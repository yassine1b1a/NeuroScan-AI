"""
Brain Tumor MRI Classification Server
======================================
Flask API server that loads a pre-trained EfficientNetB0 model
and classifies uploaded MRI images into 4 categories:
  - glioma
  - meningioma
  - pituitary
  - notumor

Usage:
  1. Make sure you have the trained model file: brain_tumor_model.keras
     (or brain_tumor_model.h5) in the same directory as this script.
  2. Install dependencies:
       pip install flask flask-cors tensorflow opencv-python numpy
  3. Run the server:
       python server.py
  4. Open index.html in your browser (or serve it with a static server).

The server exposes:
  POST /predict   — accepts multipart/form-data with field "image"
                    returns JSON with prediction results
  GET  /health    — health-check endpoint
"""

import os
import io
import time
import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend (any origin)

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE   = 150          # Must match the size used during training
MAX_BYTES  = 16 * 1024 * 1024  # 16 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

LABELS = ["glioma", "meningioma", "pituitary", "notumor"]

# Friendly display names & descriptions shown in the UI
LABEL_META = {
    "glioma": {
        "display": "Glioma Tumor",
        "description": (
            "Gliomas arise from glial cells in the brain or spine. "
            "They include types such as glioblastoma multiforme (GBM) and astrocytomas, "
            "and range from low-grade (slow-growing) to high-grade (aggressive)."
        ),
        "severity": "high",
        "color": "#ef4444",
    },
    "meningioma": {
        "display": "Meningioma Tumor",
        "description": (
            "Meningiomas form in the meninges — the protective layers surrounding "
            "the brain and spinal cord. Most are benign and slow-growing, "
            "though some can recur after treatment."
        ),
        "severity": "medium",
        "color": "#f97316",
    },
    "pituitary": {
        "display": "Pituitary Tumor",
        "description": (
            "Pituitary tumors develop in the pituitary gland at the base of the brain. "
            "They often affect hormone regulation and are usually non-cancerous (adenomas), "
            "but can cause systemic effects."
        ),
        "severity": "medium",
        "color": "#eab308",
    },
    "notumor": {
        "display": "No Tumor Detected",
        "description": (
            "The MRI scan shows no detectable tumor. "
            "The brain tissue appears within normal parameters based on the classification model."
        ),
        "severity": "none",
        "color": "#22c55e",
    },
}

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL = None

def load_model():
    """Try to load the trained Keras model from disk (lazy, on first request)."""
    global MODEL
    if MODEL is not None:
        return MODEL

    # Look for model files in order of preference
    candidates = [
        "brain_tumor_model.keras",
        "brain_tumor_model.h5",
        "effnet.h5",
    ]

    # Also search in the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = []
    for name in candidates:
        search_paths.append(name)
        search_paths.append(os.path.join(script_dir, name))

    # Late import so the server still starts even without TF installed
    try:
        import tensorflow as tf
    except ImportError:
        log.error("TensorFlow is not installed. Run: pip install tensorflow")
        raise

    for path in search_paths:
        if os.path.exists(path):
            log.info("Loading model from: %s", path)
            MODEL = tf.keras.models.load_model(path)
            log.info("Model loaded successfully.")
            return MODEL

    raise FileNotFoundError(
        "No trained model file found. Please place one of the following in the "
        "same directory as server.py:\n  " + "\n  ".join(candidates)
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Decode bytes → BGR image → resize → add batch dim."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(img, axis=0)  # shape: (1, 150, 150, 3)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Brain Tumor Classification API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()

    # ── Validate request ──────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image field in the request. Send the file under the key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename. Please select an image."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415

    # ── Read & preprocess ─────────────────────────────────────────────────────
    try:
        file_bytes = file.read()
        img_input  = preprocess_image(file_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        log.exception("Preprocessing failed")
        return jsonify({"error": f"Image preprocessing failed: {exc}"}), 500

    # ── Load model & predict ──────────────────────────────────────────────────
    try:
        model = load_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        log.exception("Model loading failed")
        return jsonify({"error": f"Could not load model: {exc}"}), 503

    try:
        raw_preds = model.predict(img_input, verbose=0)[0]   # shape: (4,)
    except Exception as exc:
        log.exception("Inference failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    # ── Build response ────────────────────────────────────────────────────────
    predicted_index = int(np.argmax(raw_preds))
    predicted_label = LABELS[predicted_index]
    confidence      = float(raw_preds[predicted_index])

    all_probs = [
        {
            "label":      label,
            "display":    LABEL_META[label]["display"],
            "probability": round(float(prob) * 100, 2),
            "color":      LABEL_META[label]["color"],
        }
        for label, prob in zip(LABELS, raw_preds)
    ]
    # Sort descending by probability for easy display
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    log.info(
        "Prediction: %-12s  confidence: %5.1f%%  elapsed: %sms",
        predicted_label, confidence * 100, elapsed_ms,
    )

    return jsonify({
        "prediction": {
            "label":       predicted_label,
            "display":     LABEL_META[predicted_label]["display"],
            "description": LABEL_META[predicted_label]["description"],
            "severity":    LABEL_META[predicted_label]["severity"],
            "color":       LABEL_META[predicted_label]["color"],
            "confidence":  round(confidence * 100, 2),
        },
        "all_probabilities": all_probs,
        "elapsed_ms": elapsed_ms,
        "model_info": {
            "architecture": "EfficientNetB0",
            "image_size":   IMG_SIZE,
            "classes":      len(LABELS),
        },
    })


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  Brain Tumor MRI Classification API")
    log.info("  Model: EfficientNetB0 (transfer learning)")
    log.info("  Classes: %s", ", ".join(LABELS))
    log.info("=" * 60)
    log.info("Starting server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
