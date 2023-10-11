# Use nvcr.io/nvidia/tensorflow:23.08-tf2-py3 as base
FROM nvcr.io/nvidia/tensorflow:23.09-tf2-py3

# Add your customizations here
# For example, to install a package with apt-get:
# RUN apt-get update && apt-get install -y \

# For Python packages, use pip:
RUN pip install --no-cache-dir numpy opencv-python-headless scikit-image pillow matplotlib imblearn
