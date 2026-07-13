# Use slim Python image as base
FROM python:3.11-slim

# Install system dependencies required by MediaPipe and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install dependencies
# Install opencv-python-headless FIRST to prevent mediapipe from pulling in
# the GUI version which requires libGL and display libraries
COPY requirements.txt .
RUN pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Render sets the PORT environment variable — bind gunicorn to it
# Default to 5000 for local testing
ENV PORT=5000

# Start gunicorn server bound to the dynamic PORT
CMD gunicorn app.server:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
