"""
Model Training Module for Brain Tumor Detection System.
Handles loading the pre-trained YOLOv8 segmentation architecture,
configuring CPU execution parameters, running model training, and saving weights.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

# Add parent directory to path to allow importing from src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    BASE_MODEL_PATH,
    DATASET_YAML,
    EPOCHS,
    IMAGE_SIZE,
    BATCH_SIZE,
    WORKERS,
    DEVICE,
    TRAINED_MODEL_DIR
)


class ModelTrainer:
    """
    Encapsulates the YOLOv8 Segmentation model training lifecycle.
    """

    def __init__(self):
        """
        Initializes the trainer by loading the base YOLOv8 segmentation model.
        """
        print("Initializing Model Trainer...")
        print(f"Dataset YAML Path: {DATASET_YAML}")
        print(f"Base Model Path: {BASE_MODEL_PATH}")
        print(f"Target Output Directory: {TRAINED_MODEL_DIR}")
        
        # Ensure output directories exist
        TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        BASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Load YOLOv8 Nano Segmentation Model directly from the local base path
        # This prevents internet connection timeouts
        self.model = YOLO(str(BASE_MODEL_PATH))

    def train(self):
        """
        Executes model training using CPU optimization parameters.
        """
        print("\nStarting YOLOv8 Segmentation Model Training on CPU...")
        print("This process will segment exact tumor boundaries without bounding boxes.\n")
        
        try:
            results = self.model.train(
                data=str(DATASET_YAML),
                epochs=EPOCHS,
                imgsz=IMAGE_SIZE,
                batch=BATCH_SIZE,
                workers=WORKERS,
                device=DEVICE,
                project=str(TRAINED_MODEL_DIR.parent),
                name="trained",
                exist_ok=True,
                plots=True,
                save=True
            )
            print("\nTraining completed successfully!")
            return results
        except Exception as e:
            print(f"\nError occurred during model training: {str(e)}")
            raise e


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()