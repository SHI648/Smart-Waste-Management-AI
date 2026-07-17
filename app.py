import os
import uuid
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, current_app
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.utils import secure_filename

from app.ml.predict import predict_image, predict_frame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "waste_management.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "app", "templates"),
    static_folder=os.path.join(BASE_DIR, "app", "static"),
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + DATABASE_PATH
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = ALLOWED_EXTENSIONS
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

db = SQLAlchemy(app)


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "image_name": self.image_name,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


with app.app_context():
    db.create_all()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def generate_unique_filename(original_filename):
    ext = original_filename.rsplit(".", 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"


def save_uploaded_file(file_storage):
    if not file_storage or file_storage.filename == "":
        raise ValueError("No file selected")

    filename = secure_filename(file_storage.filename)
    if not allowed_file(filename):
        raise ValueError("Unsupported file type. Allowed types: " + ", ".join(app.config["ALLOWED_EXTENSIONS"]))

    unique_filename = generate_unique_filename(filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file_storage.save(file_path)

    return unique_filename, file_path


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_predict():
    try:
        if "file" not in request.files:
            raise ValueError("No 'file' field found in request")

        file_storage = request.files["file"]
        filename, file_path = save_uploaded_file(file_storage)

        result = predict_image(file_path)
        if not result.get("success"):
            return jsonify(result), 422

        prediction = Prediction(
            image_name=filename,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
        )
        db.session.add(prediction)
        db.session.commit()

        result["id"] = prediction.id
        result["image_name"] = filename
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as exc:
        db.session.rollback()
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/camera", methods=["GET"])
def camera_page():
    return render_template("camera.html")


@app.route("/camera/predict", methods=["POST"])
def camera_predict():
    try:
        if "frame" not in request.files:
            raise ValueError("No 'frame' field found in request")

        file_storage = request.files["frame"]
        file_bytes = file_storage.read()
        if not file_bytes:
            raise ValueError("Empty image data received")

        np_array = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode camera frame")

        result = predict_frame(frame)
        if not result.get("success"):
            return jsonify(result), 422

        image_name = f"camera_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
        prediction = Prediction(
            image_name=image_name,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
        )
        db.session.add(prediction)
        db.session.commit()

        result["id"] = prediction.id
        result["image_name"] = image_name
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as exc:
        db.session.rollback()
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/history")
def history_page():
    try:
        predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
        return render_template("history.html", predictions=predictions)
    except Exception as exc:
        return render_template("history.html", predictions=[], error=str(exc))


@app.route("/api/history")
def api_history():
    try:
        predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
        return jsonify({"success": True, "data": [p.to_dict() for p in predictions]}), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/stats")
def api_stats():
    try:
        total_predictions = Prediction.query.count()

        class_counts = (
            db.session.query(Prediction.predicted_class, func.count(Prediction.id))
            .group_by(Prediction.predicted_class)
            .all()
        )

        average_confidence = db.session.query(func.avg(Prediction.confidence)).scalar() or 0.0

        return jsonify({
            "success": True,
            "total_predictions": total_predictions,
            "class_distribution": {cls: count for cls, count in class_counts},
            "average_confidence": round(float(average_confidence), 2),
        }), 200

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Resource not found"}), 404


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"success": False, "error": "Uploaded file is too large"}), 413


@app.errorhandler(500)
def server_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)