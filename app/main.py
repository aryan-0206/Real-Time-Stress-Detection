import cv2
import mediapipe as mp
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe import Image as MPImage
from mediapipe import ImageFormat
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Stress metric thresholds
STRESS_THRESHOLDS = {
    'calm': 0.08,   # Much lower calm threshold
    'mild': 0.15,  # Much lower mild threshold
    'high': 0.25   # Much lower high threshold
}


# Facial landmark indices for features (MediaPipe FaceLandmarker)
LIP_INDICES = [61, 291]
EYEBROW_INDICES = [70, 300]
HEAD_INDICES = [10, 152]
EYE_INDICES = [159, 145, 386, 374]


class StressDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Download the face_landmarker.task from mediapipe github or use a local copy
        import os
        self.model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        print(f"[DEBUG] Looking for model at: {self.model_path}")
        try:
            self.face_landmarker = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=base_options.BaseOptions(model_asset_path=self.model_path),
                    running_mode=VisionTaskRunningMode.IMAGE,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
            )
        except Exception as e:
            print(f"Error loading FaceLandmarker model at {self.model_path}. Make sure 'face_landmarker.task' is in the app folder.")
            raise e
        self.prev_blinks = 0
        self.blink_count = 0
        self.prev_eye_state = 1
        self.frame_count = 0
        self.blink_rate = 0
        # Remove hand detection, only use face
        self.hand_on_head_timer = 0
        self.last_hand_on_head_time = 0

    def get_metrics(self, landmarks, img_w, img_h):
        # Eyebrow raise (vertical distance between eyebrow and eye)
        brow = np.array([landmarks[EYEBROW_INDICES[0]].x * img_w, landmarks[EYEBROW_INDICES[0]].y * img_h])
        eye = np.array([landmarks[EYE_INDICES[0]].x * img_w, landmarks[EYE_INDICES[0]].y * img_h])
        eyebrow_raise = np.linalg.norm(brow - eye) / img_h

        # Lip tension (distance between lip corners)
        lip_left = np.array([landmarks[LIP_INDICES[0]].x * img_w, landmarks[LIP_INDICES[0]].y * img_h])
        lip_right = np.array([landmarks[LIP_INDICES[1]].x * img_w, landmarks[LIP_INDICES[1]].y * img_h])
        lip_tension = np.linalg.norm(lip_left - lip_right) / img_w

        # Head nod (vertical distance between chin and forehead)
        chin = np.array([landmarks[HEAD_INDICES[1]].x * img_w, landmarks[HEAD_INDICES[1]].y * img_h])
        forehead = np.array([landmarks[HEAD_INDICES[0]].x * img_w, landmarks[HEAD_INDICES[0]].y * img_h])
        head_nod = np.abs(chin[1] - forehead[1]) / img_h

        # Symmetry (difference between left and right facial points)
        symmetry = np.abs(landmarks[61].y - landmarks[291].y)

        # Blink detection (eye aspect ratio)
        left_eye = [landmarks[159], landmarks[145]]
        right_eye = [landmarks[386], landmarks[374]]
        left_ear = np.linalg.norm(np.array([left_eye[0].y, left_eye[0].x]) - np.array([left_eye[1].y, left_eye[1].x]))
        right_ear = np.linalg.norm(np.array([right_eye[0].y, right_eye[0].x]) - np.array([right_eye[1].y, right_eye[1].x]))
        ear = (left_ear + right_ear) / 2
        eye_state = 1 if ear > 0.01 else 0
        if self.prev_eye_state == 1 and eye_state == 0:
            self.blink_count += 1
        self.prev_eye_state = eye_state
        self.frame_count += 1
        if self.frame_count >= 30:
            self.blink_rate = self.blink_count * 2  # per minute (approx)
            self.blink_count = 0
            self.frame_count = 0
        return eyebrow_raise, lip_tension, head_nod, symmetry, self.blink_rate

    def get_stress_score(self, metrics):
        # Make the score much more sensitive to facial changes
        eyebrow_raise, lip_tension, head_nod, symmetry, blink_rate = metrics
        # Use higher weights and sum all metrics
        score = (
            0.6 * eyebrow_raise +
            0.8 * lip_tension +
            0.4 * head_nod +
            0.2 * symmetry +
            0.3 * (blink_rate / 30)
        )
        return min(score, 1.0)

    def detect_hand_on_head(self, frame, face_landmarks):
        # Hand detection removed, always return False
        return False

    def release(self):
        self.cap.release()

class StressApp:
    def __init__(self, root):
        self.detector = StressDetector()
        self.root = root
        self.root.title('Real-Time Stress Detection')
        self.start_time = None
        self.high_stress_active = False
        self.high_stress_start_time = 0
        # Main container
        self.main_frame = tk.Frame(root, bg='black')
        self.main_frame.pack(fill='both', expand=True)
        # Video panel (left)
        self.video_panel = tk.Label(self.main_frame, bg='black')
        self.video_panel.grid(row=0, column=0, sticky='nsew')
        # Metrics panel (right)
        self.metrics_frame = tk.Frame(self.main_frame, bg='white', width=320)
        self.metrics_frame.grid(row=0, column=1, sticky='nsew')
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        # High stress label
        self.stress_title = Label(self.metrics_frame, font=('Arial', 20, 'bold'), bg='white', fg='red')
        self.stress_title.pack(anchor='n', pady=(16, 4))
        # Score label
        self.score_label = Label(self.metrics_frame, font=('Arial', 16, 'bold'), bg='white')
        self.score_label.pack(anchor='n', pady=(0, 8))
        # Metric bars
        self.bars = {}
        for metric in ['Eyebrow Raise', 'Lip Tension', 'Head Nod', 'Symmetry', 'Blink Rate/min']:
            bar_frame = tk.Frame(self.metrics_frame, bg='white')
            bar_frame.pack(fill='x', pady=2, padx=10)
            label = Label(bar_frame, text=metric+':', font=('Arial', 12), width=16, anchor='w', bg='white')
            label.pack(side='left')
            canvas = tk.Canvas(bar_frame, width=140, height=18, bg='white', highlightthickness=0)
            canvas.pack(side='left')
            value_label = Label(bar_frame, font=('Arial', 12), width=7, anchor='w', bg='white')
            value_label.pack(side='left')
            self.bars[metric] = (canvas, value_label)
        # Stress level label
        self.stress_label = Label(self.metrics_frame, font=('Arial', 18, 'bold'), bg='white')
        self.stress_label.pack(anchor='center', pady=12)
        # Legend
        legend_frame = tk.Frame(self.metrics_frame, bg='white')
        legend_frame.pack(anchor='s', pady=(24, 8), side='bottom')
        for color, text in [("#43a047", "Calm"), ("#fbc02d", "Mild"), ("#e53935", "High")]:
            dot = tk.Canvas(legend_frame, width=16, height=16, bg='white', highlightthickness=0)
            dot.create_oval(2, 2, 14, 14, fill=color, outline='')
            dot.pack(side='left', padx=2)
            Label(legend_frame, text=text, font=('Arial', 12), bg='white').pack(side='left', padx=(0, 8))
        # Timer overlay
        self.timer_label = tk.Label(self.video_panel, font=('Arial', 16, 'bold'), fg='white', bg='black')
        self.timer_label.place(relx=0.95, rely=0.02, anchor='ne')
        self.update_frame()
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)


    def update_frame(self):
        import time
        ret, frame = self.detector.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=img_rgb)
        result = self.detector.face_landmarker.detect(mp_image)
        stress_level = 'Calm'
        color = 'green'
        score = 0
        metrics = (0, 0, 0, 0, 0)
        show_landmarks = False
        now = time.time()
        if result and result.face_landmarks and len(result.face_landmarks) > 0:
            face_landmarks = result.face_landmarks[0]
            img_h, img_w, _ = frame.shape
            metrics = self.detector.get_metrics(face_landmarks, img_w, img_h)
            score = self.detector.get_stress_score(metrics)
            show_landmarks = True
            # Only draw a subset of landmarks: jawline, eyebrows, eyes, mouth corners
            key_indices = [10, 152, 234, 454, 61, 291, 70, 300, 159, 145, 386, 374]  # chin, forehead, jaw, mouth, eyebrows, eyes
            for idx in key_indices:
                lm = face_landmarks[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

            # --- Improved 5-second stress level logic (face only) ---
            if not hasattr(self, 'last_stress_level'):
                self.last_stress_level = 'Calm'
                self.last_stress_time = now
            # Determine new level
            if score >= STRESS_THRESHOLDS['high']:
                new_level = 'High'
                new_color = 'red'
            elif score >= STRESS_THRESHOLDS['mild']:
                new_level = 'Mild'
                new_color = 'yellow'
            else:
                new_level = 'Calm'
                new_color = 'green'
            # If level changes, update timer and last level
            if new_level != self.last_stress_level:
                self.last_stress_time = now
                self.last_stress_level = new_level
                self.last_stress_color = new_color
            # Always show last detected level for 5 seconds
            if now - self.last_stress_time < 5:
                stress_level = self.last_stress_level
                color = self.last_stress_color
            else:
                stress_level = 'Calm'
                color = 'green'
        else:
            self.last_stress_level = None
        # Draw a red border if high stress
        if stress_level == 'High':
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (40, 40, 200), 8)
        # Timer overlay (top right)
        if self.start_time is None:
            self.start_time = now
        elapsed = int(now - self.start_time)
        timer_text = f"{elapsed//60:02d}:{elapsed%60:02d}"
        cv2.rectangle(frame, (frame.shape[1]-90, 10), (frame.shape[1]-10, 50), (0,0,0), -1)
        cv2.putText(frame, timer_text, (frame.shape[1]-80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Convert to ImageTk
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_panel.imgtk = imgtk
        self.video_panel.configure(image=imgtk)
        # Metrics panel update
        eyebrow_raise, lip_tension, head_nod, symmetry, blink_rate = metrics
        self.score_label.config(text=f"Score: {score:.2f}")
        bar_data = [
            ("Eyebrow Raise", eyebrow_raise, 0.2),
            ("Lip Tension", lip_tension, 1.0),
            ("Head Nod", head_nod, 0.5),
            ("Symmetry", symmetry, 0.2),
            ("Blink Rate/min", blink_rate, 30.0)
        ]
        for metric, value, maxv in bar_data:
            canvas, value_label = self.bars[metric]
            canvas.delete('all')
            frac = min(value / maxv, 1.0)
            if frac > 0.7:
                bar_color = '#e53935'  # red
            elif frac > 0.4:
                bar_color = '#fbc02d'  # yellow
            else:
                bar_color = '#43a047'  # green
            canvas.create_rectangle(0, 2, int(120*frac), 16, fill=bar_color, outline='')
            canvas.create_rectangle(0, 2, 120, 16, outline='#888')
            value_label.config(text=f"{value:.3f}")
        # Stress level label and title
        if stress_level == 'High':
            self.stress_title.config(text="HIGH STRESS", fg='#e53935')
            stress_fg = '#fff'
            stress_bg = '#e53935'
        elif stress_level == 'Mild':
            self.stress_title.config(text="MILD STRESS", fg='#fbc02d')
            stress_fg = '#222'
            stress_bg = '#fbc02d'
        else:
            self.stress_title.config(text="CALM", fg='#43a047')
            stress_fg = '#fff'
            stress_bg = '#43a047'
        self.stress_label.config(text=f"Stress: {stress_level}", fg=stress_fg, bg=stress_bg)
        self.root.after(33, self.update_frame)

    def on_close(self):
        self.detector.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = StressApp(root)
    root.mainloop()
