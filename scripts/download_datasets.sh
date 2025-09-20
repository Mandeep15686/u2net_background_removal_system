#!/bin/bash
# Dataset Download and Preparation Script
# Team 1: The Isolationists - U²-Net Background Removal System

echo "U²-Net Dataset Download and Preparation"
echo "======================================"

# Create data directories
mkdir -p data/datasets/DUTS/{DUTS-TR,DUTS-TE}/{DUTS-TR-Image,DUTS-TR-Mask,DUTS-TE-Image,DUTS-TE-Mask}
mkdir -p data/datasets/HKU-IS/{imgs,gt}
mkdir -p data/datasets/PASCAL-S/{imgs,gt}
mkdir -p data/sample_images

echo "Directories created successfully!"

echo "Dataset URLs (manual download required):"
echo "----------------------------------------"
echo "DUTS: http://saliencydetection.net/duts/"
echo "HKU-IS: https://i.cs.hku.hk/~gbli/deep_saliency.html"
echo "PASCAL-S: http://cbs.ic.gatech.edu/salobj/"
echo ""
echo "Please download datasets manually and extract to data/datasets/"
echo ""
echo "Expected structure:"
echo "data/datasets/"
echo "├── DUTS/"
echo "│   ├── DUTS-TR/"
echo "│   │   ├── DUTS-TR-Image/"
echo "│   │   └── DUTS-TR-Mask/"
echo "│   └── DUTS-TE/"
echo "│       ├── DUTS-TE-Image/"
echo "│       └── DUTS-TE-Mask/"
echo "├── HKU-IS/"
echo "│   ├── imgs/"
echo "│   └── gt/"
echo "└── PASCAL-S/"
echo "    ├── imgs/"
echo "    └── gt/"
