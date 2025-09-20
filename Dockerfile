
# U²-Net Background Removal API Docker Image
# Team 1: The Isolationists - Production Deployment

FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models checkpoints logs cache

# Install the package
RUN pip3 install -e .

# Set environment variables for the application
ENV MODEL_PATH=/app/models/u2net_best.pth
ENV CONFIG_PATH=/app/configs/deployment_config.json
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create non-root user
RUN useradd -m -u 1000 u2net && chown -R u2net:u2net /app
USER u2net

# Run the application
CMD ["python3", "-m", "src.api.deployment", "--model-path", "/app/models/u2net_best.pth", "--config", "production", "--host", "0.0.0.0", "--port", "8000"]
