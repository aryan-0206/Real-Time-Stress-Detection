"""
server.py — Flask web server for Real-Time Stress Detection.
Accepts base64-encoded webcam frames, runs MediaPipe analysis, returns JSON.
"""

import os
import base64
import io
import logging

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

from detector import StressDetector

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")
CORS(app)

# Initialise the detector once at startup (loads the model)
logger.info("Loading StressDetector model…")
detector = StressDetector()
logger.info("StressDetector ready.")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Render health-check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts JSON: { "image": "<base64-encoded JPEG/PNG>" }
    Returns JSON stress analysis results.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in JSON body"}), 400

    # Decode base64 → PIL Image → numpy RGB array
    try:
        image_b64 = data["image"]
        # Strip data-URL header if present (data:image/jpeg;base64,...)
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb_array   = np.array(pil_image, dtype=np.uint8)
    except Exception as exc:
        logger.warning("Failed to decode image: %s", exc)
        return jsonify({"error": f"Invalid image data: {exc}"}), 422

    # Run stress detection
    try:
        result = detector.analyze_frame(rgb_array)
    except Exception as exc:
        logger.error("Detection error: %s", exc, exc_info=True)
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    return jsonify(result)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False is important for production; Render uses gunicorn anyway
    app.run(host="0.0.0.0", port=port, debug=False)
