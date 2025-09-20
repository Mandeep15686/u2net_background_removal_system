#!/bin/bash
# API Server Launcher
# Team 1: The Isolationists - U²-Net API Deployment

set -e

echo "U²-Net API Server Launcher"
echo "========================="

# Default parameters
MODEL_PATH="models/u2net_best.pth"
CONFIG="development"
HOST="0.0.0.0"
PORT=8000
WORKERS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path MODEL_PATH Model file path (default: models/u2net_best.pth)"
            echo "  --config CONFIG         Configuration (development/production, default: development)"
            echo "  --host HOST             Host address (default: 0.0.0.0)"
            echo "  --port PORT             Port number (default: 8000)"
            echo "  --workers WORKERS       Number of workers (default: 1)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "The API will start but inference may fail without a trained model."
fi

echo "API Configuration:"
echo "-----------------"
echo "Model path: $MODEL_PATH"
echo "Config: $CONFIG"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo ""

echo "Starting API server..."
echo "Server will be available at: http://$HOST:$PORT"
echo "Interactive docs at: http://$HOST:$PORT/docs"
echo ""

# Start API server
python -m src.api.deployment \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS"
