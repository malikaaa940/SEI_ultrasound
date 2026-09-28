# SEI Ultrasound Segmentation & Quality Assessment

This repository contains a machine learning pipeline for the multi-class segmentation of ultrasound images and the subsequent evaluation of scan quality. Built with Python and PyTorch, it leverages the Ultralytics YOLOv8 framework to parse critical anatomical structures and generate automated, expert-informed quality scores.

## Key Repository Files

The repository is structured to take ultrasound data from raw annotations through to a fully evaluated segmentation model. Here is a breakdown of the most important files driving the project:

### 1. `dataset.yaml`
This configuration file is the core map for the model's data. It defines the paths to the training, validation, and testing splits, and specifically maps the four critical anatomical classes targeted by the segmentation model:
*   `abdomen`
*   `stomach`
*   `umbilical_vein`
*   `ribs`

### 2. `train_segmentation.py`
This script handles the initialization and training loop for the **YOLOv8n-seg** model. It sets up the PyTorch environment configured to utilize Apple Silicon via the `mps` (Metal Performance Shaders) backend for hardware acceleration. It manages hyperparameters, handles the data loading of the ultrasound imagery, and exports the fine-tuned model weights upon completion.

### 3. `quality_assessment.py`
This file implements the semi-automatic quality assessment system. Once the segmentation model processes a scan, this script takes the inference data and calculates stringent evaluation metrics—most notably **mAP50** (mean Average Precision at 50% IoU). By evaluating the model's accuracy in identifying the required anatomical components, the script assigns a formalized quality score to the scan.

### 4. `inference.py`
Designed for testing and demonstration, this script loads the pre-trained YOLO model and runs new ultrasound images through the pipeline. It outputs the visualized segmentation masks overlaid onto the original scans to physically verify the model's accuracy on the abdomen, stomach, vein, and ribs.

### 5. `requirements.txt`
Contains the necessary environment dependencies to execute the scripts, relying heavily on `torch`, `torchvision`, and `ultralytics`.

---

## Hardware Setup
The deep learning operations in this repository are optimized for macOS environments running Apple Silicon (such as M2 chips). The PyTorch tensors and model weights are explicitly directed to the MPS device to accelerate both training and inference times without requiring a dedicated Nvidia GPU.

## Project Objective
By successfully isolating specific anatomical features, this system provides a reliable, automated scoring mechanism for ultrasound scans. It ensures that collected imagery meets necessary quality thresholds by verifying the presence and clarity of vital physiological landmarks.
