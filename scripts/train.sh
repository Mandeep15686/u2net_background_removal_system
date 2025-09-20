#!/bin/bash
# Training Script Launcher
# Team 1: The Isolationists - U²-Net Training

set -e

echo "U²-Net Training Launcher"
echo "======================"

# Default parameters
CONFIG="configs/training_config.json"
EPOCHS=100
BATCH_SIZE=8
LR=0.001
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config CONFIG_FILE    Training configuration file (default: configs/training_config.json)"
            echo "  --epochs EPOCHS         Number of training epochs (default: 100)"
            echo "  --batch-size BATCH_SIZE Batch size (default: 8)"
            echo "  --lr LEARNING_RATE      Learning rate (default: 0.001)"
            echo "  --device DEVICE         Device to use (default: cuda)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if datasets exist
if [ ! -d "data/datasets/DUTS" ]; then
    echo "Error: DUTS dataset not found. Please run scripts/download_datasets.sh first."
    exit 1
fi

# Create checkpoints directory
mkdir -p checkpoints logs

echo "Training Configuration:"
echo "----------------------"
echo "Config file: $CONFIG"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Device: $DEVICE"
echo ""

# Start training
echo "Starting training..."
python -m src.training.train \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --save-dir checkpoints

echo "Training completed!"
