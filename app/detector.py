"""
detector.py — Core stress detection logic (no GUI, no webcam).
Used by the Flask server to analyze individual video frames.
"""

import os
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe import Image as MPImage
from mediapipe import ImageFormat

# ── Stress metric thresholds ─────────────────────────────────────────────────
STRESS_THRESHOLDS = {
    'calm': 0.08,
    'mild': 0.15,
    'high': 0.25,
}

# ── Facial landmark indices (MediaPipe FaceLandmarker 478-point model) ────────
LIP_INDICES      = [61, 291]
EYEBROW_INDICES  = [70, 300]
HEAD_INDICES     = [10, 152]
EYE_INDICES      = [159, 145, 386, 374]


class StressDetector:
    """Stateful stress detector.  Call analyze_frame() per frame."""

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"face_landmarker.task not found at {model_path}. "
                "Download it from https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
        print(f"[StressDetector] Loading model from: {model_path}")
        with open(model_path, "rb") as f:
            model_data = f.read()

        self.face_landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_buffer=model_data),
                running_mode=VisionTaskRunningMode.IMAGE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
        )
        # Blink tracking state
        self.prev_eye_state = 1
        self.blink_count    = 0
        self.frame_count    = 0
        self.blink_rate     = 0  # estimated blinks/minute

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_metrics(self, landmarks, img_w: int, img_h: int):
        """Compute raw facial metrics from landmark list."""

        # Eyebrow raise — vertical distance eyebrow → eye / image height
        brow = np.array([landmarks[EYEBROW_INDICES[0]].x * img_w,
                         landmarks[EYEBROW_INDICES[0]].y * img_h])
        eye  = np.array([landmarks[EYE_INDICES[0]].x * img_w,
                         landmarks[EYE_INDICES[0]].y * img_h])
        eyebrow_raise = np.linalg.norm(brow - eye) / img_h

        # Lip tension — horizontal distance between lip corners / image width
        lip_left  = np.array([landmarks[LIP_INDICES[0]].x * img_w,
                               landmarks[LIP_INDICES[0]].y * img_h])
        lip_right = np.array([landmarks[LIP_INDICES[1]].x * img_w,
                               landmarks[LIP_INDICES[1]].y * img_h])
        lip_tension = np.linalg.norm(lip_left - lip_right) / img_w

        # Head nod — vertical extent chin → forehead / image height
        chin     = np.array([landmarks[HEAD_INDICES[1]].x * img_w,
                              landmarks[HEAD_INDICES[1]].y * img_h])
        forehead = np.array([landmarks[HEAD_INDICES[0]].x * img_w,
                              landmarks[HEAD_INDICES[0]].y * img_h])
        head_nod = np.abs(chin[1] - forehead[1]) / img_h

        # Symmetry — vertical discrepancy between mirrored lip landmarks
        symmetry = float(np.abs(landmarks[61].y - landmarks[291].y))

        # Blink detection — simple eye aspect ratio
        l0, l1 = landmarks[159], landmarks[145]
        r0, r1 = landmarks[386], landmarks[374]
        left_ear  = abs(l0.y - l1.y)
        right_ear = abs(r0.y - r1.y)
        ear = (left_ear + right_ear) / 2.0
        eye_state = 1 if ear > 0.01 else 0
        if self.prev_eye_state == 1 and eye_state == 0:
            self.blink_count += 1
        self.prev_eye_state = eye_state
        self.frame_count += 1
        if self.frame_count >= 30:
            self.blink_rate  = self.blink_count * 2   # approx per-minute
            self.blink_count = 0
            self.frame_count = 0

        return eyebrow_raise, lip_tension, head_nod, symmetry, self.blink_rate

    def _stress_score(self, metrics) -> float:
        eyebrow_raise, lip_tension, head_nod, symmetry, blink_rate = metrics
        score = (
            0.6 * eyebrow_raise +
            0.8 * lip_tension   +
            0.4 * head_nod      +
            0.2 * symmetry      +
            0.3 * (blink_rate / 30.0)
        )
        return float(min(score, 1.0))

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_frame(self, rgb_array: np.ndarray) -> dict:
        """
        Analyze a single RGB image (H×W×3 uint8 numpy array).

        Returns a dict:
          face_detected  : bool
          stress_level   : "Calm" | "Mild" | "High"
          score          : float 0–1
          eyebrow_raise  : float
          lip_tension    : float
          head_nod       : float
          symmetry       : float
          blink_rate     : float (approx per minute)
          landmarks      : list of {x, y} dicts for the 12 key points (for overlay)
        """
        img_h, img_w = rgb_array.shape[:2]
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_array)
        result   = self.face_landmarker.detect(mp_image)

        if not (result and result.face_landmarks and len(result.face_landmarks) > 0):
            return {
                "face_detected": False,
                "stress_level":  "Calm",
                "score":         0.0,
                "eyebrow_raise": 0.0,
                "lip_tension":   0.0,
                "head_nod":      0.0,
                "symmetry":      0.0,
                "blink_rate":    0.0,
                "landmarks":     [],
            }

        face_landmarks = result.face_landmarks[0]
        metrics        = self._get_metrics(face_landmarks, img_w, img_h)
        score          = self._stress_score(metrics)

        if score >= STRESS_THRESHOLDS['high']:
            stress_level = "High"
        elif score >= STRESS_THRESHOLDS['mild']:
            stress_level = "Mild"
        else:
            stress_level = "Calm"

        # Key landmark positions (normalised 0–1) for browser overlay
        key_indices = [10, 152, 234, 454, 61, 291, 70, 300, 159, 145, 386, 374]
        landmarks_out = [
            {"x": face_landmarks[i].x, "y": face_landmarks[i].y}
            for i in key_indices
        ]

        eyebrow_raise, lip_tension, head_nod, symmetry, blink_rate = metrics
        return {
            "face_detected": True,
            "stress_level":  stress_level,
            "score":         round(score, 4),
            "eyebrow_raise": round(float(eyebrow_raise), 4),
            "lip_tension":   round(float(lip_tension), 4),
            "head_nod":      round(float(head_nod), 4),
            "symmetry":      round(float(symmetry), 4),
            "blink_rate":    round(float(blink_rate), 2),
            "landmarks":     landmarks_out,
        }
