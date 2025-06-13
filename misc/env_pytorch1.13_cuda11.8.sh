#!/bin/bash

# Environment name
ENV_NAME="camoGAN_py39pt113cu118" # Short and descriptive camo_pt113_cu118
PYTHON_VERSION="3.9" # Python 3.7 was original, 3.9 is a safe upgrade for newer libs

echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -n $ENV_NAME -y python=$PYTHON_VERSION
conda activate $ENV_NAME

echo "Installing PyTorch 1.13.1 with CUDA 11.8 support..."
# Attempting CUDA 11.8 as requested. PyTorch 1.13.1, torchvision 0.14.1, torchaudio 0.13.1
# If this specific combination is not found, an alternative might be PyTorch 1.13.0 with CUDA 11.8
# or PyTorch 1.13.1 with CUDA 11.7.
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.8 -c pytorch -c nvidia

echo "Installing PyTorch3D dependencies (fvcore, iopath)..."
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath

echo "Installing PyTorch3D (targeting version compatible with PyTorch 1.13.1)..."
# PyTorch3D v0.7.4 is known to be compatible with PyTorch 1.13.1.
# The original project used v0.4.0, which is likely too old for PyTorch 1.13.1.
pip install "pytorch3d==0.7.4"

echo "Installing Kornia..."
# Sticking to kornia==0.5.0 as in the original env.sh to minimize changes.
# This version is older; if it causes issues with PyTorch 1.13.1,
# consider upgrading to a newer Kornia (e.g., pip install "kornia>=0.7.0").
pip install "kornia==0.5.0"

echo "Installing other dependencies..."
# From original env.sh, using scikit-learn explicitly
pip install open3d pandas opencv-python trimesh[easy] matplotlib scikit-learn scikit-image tensorboardX tqdm lpips

echo "Installing pyembree from conda-forge..."
conda install -y -c conda-forge pyembree

echo "Installing pytorch-gradual-warmup-lr from git..."
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

echo "-------------------------------------------------------------------"
echo "Environment $ENV_NAME setup is complete."
echo "To activate this environment, run: conda activate $ENV_NAME"
echo "If you encounter issues (e.g., PyTorch with CUDA 11.8, Kornia 0.5.0, or PyTorch3D):"
echo "1. For PyTorch: Try pytorch-cuda=11.7 if 11.8 fails, or PyTorch 1.13.0."
echo "2. For Kornia: If kornia 0.5.0 fails, try 'pip install kornia' for the latest or 'pip install \"kornia>=0.7.0\"'."
echo "3. For PyTorch3D: Ensure version 0.7.4 is compatible or try 'pip install pytorch3d'."
echo "-------------------------------------------------------------------"