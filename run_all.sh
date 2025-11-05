#!/bin/bash
# Complete pipeline to run all steps

echo "=============================================="
echo "LSTM Frequency Filter - Complete Pipeline"
echo "=============================================="
echo ""

# Get the Python executable path
PYTHON="/Users/bz0r7y/private/NBR/school-2025/semester-1/llm-orchestration/ex02/lstm-frequency-filter/.venv/bin/python"

echo "Step 1: Generating dataset..."
$PYTHON generate_dataset.py
if [ $? -ne 0 ]; then
    echo "Error in step 1"
    exit 1
fi
echo ""

echo "Step 2: Creating data visualizations..."
$PYTHON visualize_data.py
if [ $? -ne 0 ]; then
    echo "Error in step 2"
    exit 1
fi
echo ""

echo "Step 3: Preparing training data..."
$PYTHON prepare_training_data.py
if [ $? -ne 0 ]; then
    echo "Error in step 3"
    exit 1
fi
echo ""

echo "Step 4: Training LSTM model..."
$PYTHON train_model.py
if [ $? -ne 0 ]; then
    echo "Error in step 4"
    exit 1
fi
echo ""

echo "Step 5: Evaluating model..."
$PYTHON evaluate_model.py
if [ $? -ne 0 ]; then
    echo "Error in step 5"
    exit 1
fi
echo ""

echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Dataset: data/"
echo "  - Model: models/best_model.pth"
echo "  - Visualizations: visualizations/"
echo ""
echo "Total visualizations created: 13"
echo ""
