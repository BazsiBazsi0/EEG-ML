# Use nvcr.io/nvidia/tensorflow:23.08-tf2-py3 as base
FROM nvcr.io/nvidia/tensorflow:24.06-tf2-py3

# Add your customizations here
# For example, to install a package with apt-get:
# RUN apt-get update && apt-get install -y \

# For Python packages, use pip:
RUN python -m pip install --upgrade pip
RUN install numpy==1.26.0 PyYAML imblearn mne tensorflow==2.14 coverage pandas seaborn matplotlib keras-tuner
