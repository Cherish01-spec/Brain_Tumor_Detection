"""
Configuration module for Brain Tumor Detection System.
Contains system paths, model hyperparameters, and application settings.
"""

from pathlib import Path

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Dataset Paths
DATASET_YAML = DATA_DIR / "dataset.yaml"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Configuration
# Using YOLOv8 Nano Segmentation model for precise edge segmentation and fast CPU inference
BASE_MODEL_NAME = "yolov8n-seg.pt"
BASE_MODEL_PATH = MODELS_DIR / "base" / BASE_MODEL_NAME
TRAINED_MODEL_DIR = MODELS_DIR / "trained"
BEST_MODEL_PATH = TRAINED_MODEL_DIR / "weights" / "best.pt"

# Training Hyperparameters (Optimized for Windows 11 CPU and 16 GB RAM)
EPOCHS = 25
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 2
DEVICE = "cpu"

# Class Definitions
CLASS_NAMES = {
    0: "tumor_good_chance",
    1: "tumor_less_chance",
    2: "tumor_moderate_chance"
}

# Output Paths
LOGS_DIR = OUTPUTS_DIR / "logs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"