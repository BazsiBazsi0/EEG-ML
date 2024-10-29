# Use NVIDIA's official TensorFlow container as base
FROM nvcr.io/nvidia/tensorflow:24.10-tf2-py3

# Set working directory
WORKDIR /workspace

# Create directory for dataset
RUN mkdir -p /workspace/dataset

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir \
    numpy==1.26.0 \
    PyYAML \
    imblearn \
    mne \
    coverage \
    pandas \
    seaborn \
    matplotlib \
    keras-tuner

# Copy project files, only use it if you dont mount the project directory
#COPY . .

# Set Python path
ENV PYTHONPATH=/workspace

# Default command
CMD ["python3", "main.py"]