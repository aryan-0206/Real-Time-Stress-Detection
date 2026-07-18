"""
server.py — Flask web server for Real-Time Stress Detection.
Accepts base64-encoded webcam frames, runs MediaPipe analysis, returns JSON.
"""

import os
import sys
import base64
import io
import logging

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve template directory relative to this file (works in all run modes)
_HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_HERE, "templates"))
CORS(app)

# ── Import detector ────────────────────────────────────────────────────────────
# Works whether run as `python app/server.py` or `gunicorn app.server:app`
try:
    from app.detector import StressDetector        # gunicorn from project root
except ImportError:
    from detector import StressDetector            # direct python execution

# ── Load model at startup ─────────────────────────────────────────────────────
_detector: "StressDetector | None" = None
_detector_error: "str | None" = None

logger.info("Loading StressDetector model…")
try:
    _detector = StressDetector()
    logger.info("StressDetector ready ✓")
except FileNotFoundError as exc:
    _detector_error = (
        "face_landmarker.task model file not found. "
        "Make sure it exists at app/face_landmarker.task. "
        f"Details: {exc}"
    )
    logger.error("FATAL: %s", _detector_error)
    sys.exit(1)
except Exception as exc:
    _detector_error = f"Could not initialise StressDetector: {exc}"
    logger.error("FATAL: %s", _detector_error, exc_info=True)
    sys.exit(1)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health-check endpoint — returns model status alongside ok/error."""
    if _detector is None:
        return jsonify({
            "status": "error",
            "reason": _detector_error or "Detector not initialised",
        }), 503
    return jsonify({"status": "ok", "model": "face_landmarker"}), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts JSON: { "image": "<base64-encoded JPEG/PNG>" }
    Returns JSON stress analysis results.
    """
    if _detector is None:
        return jsonify({"error": "Model not loaded"}), 503

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
        result = _detector.analyze_frame(rgb_array)
    except Exception as exc:
        logger.error("Detection error: %s", exc, exc_info=True)
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    return jsonify(result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
