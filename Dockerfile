# Use slim Python image as base
FROM python:3.11-slim

# Install system dependencies required by MediaPipe and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 5000
EXPOSE 5000

# Start gunicorn server
CMD ["gunicorn", "app.server:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120"]
