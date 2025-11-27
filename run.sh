#!/bin/bash

# Phishing Website Detection - ML Pipeline Runner
# This script executes the end-to-end machine learning pipeline

echo "========================================="
echo "Phishing Website Detection ML Pipeline"
echo "========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null
then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python"
if command -v python3 &> /dev/null
then
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Error: data/ directory not found"
    exit 1
fi

# Check if phishing.db exists
if [ ! -f "data/phishing.db" ]; then
    echo "Error: data/phishing.db not found"
    exit 1
fi

echo "Data file found: data/phishing.db"
echo ""

# Create necessary directories
echo "Creating output directories..."
mkdir -p models
mkdir -p results
echo "[OK] Directories created"
echo ""

# Run the ML pipeline
echo "========================================="
echo "Executing ML Pipeline..."
echo "========================================="
echo ""

$PYTHON_CMD src/run_pipeline.py

# Check if pipeline executed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "[SUCCESS] Pipeline completed successfully!"
    echo "========================================="
    echo ""
    echo "Generated outputs:"
    echo "  - models/phishing_detector.pkl"
    echo "  - models/preprocessor.pkl"
    echo "  - results/confusion_matrix.png"
    echo "  - results/roc_curve.png"
    echo "  - results/feature_importance.png"
    echo "  - results/evaluation_metrics.txt"
    echo ""
else
    echo ""
    echo "========================================="
    echo "[ERROR] Pipeline execution failed!"
    echo "========================================="
    exit 1
fi

