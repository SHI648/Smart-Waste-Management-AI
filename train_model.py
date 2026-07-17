import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.ml.preprocessing import (
    IMG_SIZE,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    CLASS_NAMES,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    create_data_generators,
)

MODEL_DIR = "model"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "waste_classifier.keras")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_checkpoint.keras")

LOGS_DIR = os.path.join("logs", "plots")
ACCURACY_PLOT_PATH = os.path.join(LOGS_DIR, "accuracy_curve.png")
LOSS_PLOT_PATH = os.path.join(LOGS_DIR, "loss_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(LOGS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(LOGS_DIR, "classification_report.txt")

NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

INITIAL_EPOCHS = 20
INITIAL_LEARNING_RATE = 1e-3

FINE_TUNE_EPOCHS = 10
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_AT_LAYER = 100


def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dense(128, activation="relu", name="dense_head")(x)
    x = Dropout(0.3, name="dropout_head")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="waste_classifier_mobilenetv2")

    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def get_callbacks(checkpoint_path=CHECKPOINT_PATH):
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    return [early_stopping, model_checkpoint, reduce_lr]


def train_phase_1(model, train_generator, val_generator, callbacks):
    print("\n===== PHASE 1: Training classification head (base frozen) =====")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def fine_tune_phase_2(model, base_model, train_generator, val_generator, callbacks):
    print("\n===== PHASE 2: Fine-tuning top layers of MobileNetV2 =====")

    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def combine_histories(history_1, history_2=None):
    combined = {key: list(values) for key, values in history_1.history.items()}
    if history_2 is not None:
        for key, values in history_2.history.items():
            combined.setdefault(key, [])
            combined[key].extend(values)
    return combined


def plot_accuracy_curve(history_dict, save_path=ACCURACY_PLOT_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["accuracy"], label="Training Accuracy")
    plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Accuracy curve saved to: {save_path}")


def plot_loss_curve(history_dict, save_path=LOSS_PLOT_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["loss"], label="Training Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.title("Model Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Loss curve saved to: {save_path}")


def evaluate_model(model, test_generator, class_names=CLASS_NAMES):
    print("\n===== EVALUATING ON TEST SET =====")

    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Waste Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"🧩 Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)

    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write("Smart Waste Management System - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"📄 Classification report saved to: {CLASSIFICATION_REPORT_PATH}")

    return y_true, y_pred


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("📦 Loading data generators...")
    train_generator, val_generator, test_generator = create_data_generators(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
    )

    print("🏗️  Building MobileNetV2 transfer-learning model...")
    model, base_model = build_model()
    model.summary()

    callbacks = get_callbacks()

    history_1 = train_phase_1(model, train_generator, val_generator, callbacks)
    history_2 = fine_tune_phase_2(model, base_model, train_generator, val_generator, callbacks)

    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Final model saved to: {MODEL_SAVE_PATH}")

    combined_history = combine_histories(history_1, history_2)
    plot_accuracy_curve(combined_history)
    plot_loss_curve(combined_history)

    evaluate_model(model, test_generator)

    print("\n🎉 Training pipeline complete.")


if __name__ == "__main__":
    main()