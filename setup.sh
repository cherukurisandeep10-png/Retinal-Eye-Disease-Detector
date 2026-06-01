#!/bin/bash

echo "Setting up Eye Disease Detector..."
echo ""

if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Install Python 3.10+ first."
    exit 1
fi

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Creating data directories..."
mkdir -p data/train/Normal
mkdir -p data/train/Diabetic_Retinopathy
mkdir -p data/train/Glaucoma
mkdir -p data/train/Cataract
mkdir -p data/val/Normal
mkdir -p data/val/Diabetic_Retinopathy
mkdir -p data/val/Glaucoma
mkdir -p data/val/Cataract

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Place dataset images in data/train/ and data/val/"
echo "  2. Train: cd model && python train.py"
echo "  3. Run: streamlit run app.py"
