import os
import sys
import time

import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.ml.preprocessing import (
    CLASS_NAMES,
    preprocess_single_image,
    preprocess_frame,
)

MODEL_PATH = os.path.join("model", "waste_classifier.keras")

_model_cache = None


def load_trained_model(model_path=MODEL_PATH):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Run 'python -m app.ml.train_model' first to train and save it."
        )
    return load_model(model_path)


def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_trained_model()
    return _model_cache


def get_class_probabilities(prediction_array, class_names=CLASS_NAMES):
    return {
        class_name: round(float(prediction_array[i]) * 100, 2)
        for i, class_name in enumerate(class_names)
    }


def get_predicted_class(prediction_array, class_names=CLASS_NAMES):
    predicted_index = int(np.argmax(prediction_array))
    predicted_class = class_names[predicted_index]
    confidence = round(float(prediction_array[predicted_index]) * 100, 2)
    return predicted_class, confidence


def _build_response(prediction_array, start_time, class_names=CLASS_NAMES):
    predicted_class, confidence = get_predicted_class(prediction_array, class_names)
    probabilities = get_class_probabilities(prediction_array, class_names)
    execution_time_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "success": True,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "execution_time_ms": execution_time_ms,
    }


def predict_image(image_path):
    start_time = time.time()
    try:
        model = get_model()
        preprocessed = preprocess_single_image(image_path)
        prediction_array = model.predict(preprocessed, verbose=0)[0]
        return _build_response(prediction_array, start_time)
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def predict_frame(frame):
    start_time = time.time()
    try:
        model = get_model()
        preprocessed = preprocess_frame(frame)
        prediction_array = model.predict(preprocessed, verbose=0)[0]
        return _build_response(prediction_array, start_time)
    except Exception as exc:
        return {"success": False, "error": str(exc)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.ml.predict <path_to_image>")
        sys.exit(1)

    import json

    result = predict_image(sys.argv[1])
    print(json.dumps(result, indent=2))