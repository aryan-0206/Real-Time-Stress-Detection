# Real-Time Stress Detection

AI-powered real-time stress detection using facial landmark analysis — runs entirely in your browser via a Flask/MediaPipe backend.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

---

## How it works

1. Your browser streams webcam frames to the server every 200 ms
2. The Flask server runs **MediaPipe FaceLandmarker** on each frame
3. Five facial metrics are extracted and combined into a **stress score (0–1)**
4. The UI displays live metric bars, a history chart, and a stress badge

### Stress Metrics

| Metric | Description |
|--------|-------------|
| Eyebrow Raise | Vertical distance between eyebrow and eye landmarks |
| Lip Tension | Horizontal distance between lip corner landmarks |
| Head Nod | Vertical extent from forehead to chin |
| Symmetry | Vertical discrepancy between mirrored lip landmarks |
| Blink Rate | Estimated blinks per minute |

### Stress Levels

| Level | Score threshold |
|-------|----------------|
| 🟢 Calm | < 0.08 |
| 🟡 Mild | 0.08 – 0.25 |
| 🔴 High | ≥ 0.25 |

---

## Tech Stack

- **Backend**: Python 3.11, Flask 3, Gunicorn, MediaPipe, OpenCV (headless), NumPy, Pillow
- **Frontend**: Vanilla HTML/CSS/JS, Chart.js
- **Deploy**: Docker on Render

---

## Local Development

### Prerequisites
- Python 3.11+
- Docker (optional, for container testing)

### Run with Python

```bash
# Clone the repo
git clone https://github.com/<your-username>/Real-Time-Stress-Detection.git
cd Real-Time-Stress-Detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install opencv-python-headless
pip install -r requirements.txt

# Start the dev server
python app/server.py
```

Visit `http://localhost:5000`

### Run with Docker

```bash
docker build -t stress-detection .
docker run -p 5000:10000 stress-detection
```

Visit `http://localhost:5000`

### Run tests

```bash
# With server running in another terminal:
python test_server.py
```

---

## Deploy to Render

1. Fork this repository
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` and deploy via Docker
5. The `/health` endpoint is used for health checks

The `render.yaml` is pre-configured with:
- Docker runtime (installs all system dependencies via Dockerfile)
- `PORT=10000` (Render's Docker default)
- Health check at `/health`
- Auto-deploy on every push to `main`

---

## Project Structure

```
├── app/
│   ├── detector.py          # Core stress detection logic (no GUI)
│   ├── server.py            # Flask web server
│   ├── main.py              # Desktop Tkinter app (local use only)
│   ├── face_landmarker.task # MediaPipe model file
│   └── templates/
│       └── index.html       # Frontend UI
├── Dockerfile               # Docker build for Render
├── render.yaml              # Render service configuration
├── Procfile                 # Gunicorn start command
├── requirements.txt         # Python dependencies
└── test_server.py           # End-to-end server tests
```

---

## License

MIT