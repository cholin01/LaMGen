#!/bin/bash

# 1. Create Environment (Python 3.10)
echo "🚀 Creating Conda environment: lamgen ..."
conda create -n lamgen python=3.10 -y

# 2. Activate Environment
eval "$(conda shell.bash hook)"
conda activate lamgen

# 3. Install PyTorch (Specifying official CUDA 11.8 source with +cu118 suffix)
echo "📦 Installing PyTorch 2.4.0 (CUDA 11.8)..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
echo "📦 Installing Transformers and Pandas..."
pip install transformers==4.24.0 pandas

echo "✅ Environment 'lamgen' configuration complete!"
echo "💡 To start, run: conda activate lamgen"
conda activate lamgen
