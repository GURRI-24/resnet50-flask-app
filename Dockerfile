# Use official Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install system dependencies (optional: Pillow needs libjpeg sometimes)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Render uses 10000+ internally, but 8000 is common)
EXPOSE 8000

# Command to run your Flask app (make sure app.py exists)
CMD ["python", "app.py"]
