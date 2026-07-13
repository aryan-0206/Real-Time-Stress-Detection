import base64
import io
import requests
import numpy as np
from PIL import Image

def test_health():
    print("Testing /health endpoint...")
    try:
        r = requests.get("http://127.0.0.1:5000/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert r.json() == {"status": "ok"}, f"Expected status ok, got {r.json()}"
        print("✅ /health endpoint passed!")
    except Exception as e:
        print(f"❌ /health endpoint failed: {e}")
        raise e

def test_analyze_no_face():
    print("Testing /analyze endpoint with a blank black image (no face)...")
    try:
        # Create a small blank black image (100x100 RGB)
        img = Image.new('RGB', (100, 100), color='black')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        payload = {"image": img_b64}
        r = requests.post("http://127.0.0.1:5000/analyze", json=payload)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        
        data = r.json()
        assert data["face_detected"] is False, "Expected face_detected to be False"
        assert data["stress_level"] == "Calm", f"Expected Calm, got {data['stress_level']}"
        assert data["score"] == 0.0, f"Expected score 0.0, got {data['score']}"
        print("✅ /analyze endpoint (no face) passed!")
    except Exception as e:
        print(f"❌ /analyze endpoint (no face) failed: {e}")
        raise e

if __name__ == "__main__":
    print("Running end-to-end tests for Real-Time Stress Detection server...")
    try:
        test_health()
        test_analyze_no_face()
        print("\n🎉 All tests passed successfully!")
    except Exception:
        import sys
        sys.exit(1)
