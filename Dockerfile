# Use NVIDIA CUDA 11.8 with cuDNN 8 as the base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and set alias
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch (Compatible with CUDA 11.8)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow (Required by bin2cell, older version for 2022 compatibility)
# TensorFlow 2.11 or 2.12 supports CUDA 11.8
RUN pip3 install --no-cache-dir "tensorflow<2.13"

# Install core analysis libraries
RUN pip3 install --no-cache-dir \
    scanpy \
    scvi-tools \
    bin2cell \
    seaborn \
    adjustText \
    gdown \
    plotly \
    scipy \
    scikit-learn \
    opencv-python

# Install Jupyter and Widgets for GUI support
RUN pip3 install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    ipympl

# Install your package: huetracer
# Since it's in development, we install from the provided setup.py requirements
# or directly from pip if available.
RUN pip3 install --no-cache-dir huetracer

# Set the working directory
WORKDIR /app

# Expose Jupyter port
EXPOSE 8152

# Default command: launch jupyter lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8152", "--no-browser", "--allow-root", "--NotebookApp.token=''"]