"""
Configuration for CNN + KAN Image Classification
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Support multiple datasets
TRAIN_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Chinese medicinal blossom-dataset", "ImageDatastore", "Partition", "Train")
VAL_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Chinese medicinal blossom-dataset", "ImageDatastore", "Partition", "Val")
TEST_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Chinese medicinal blossom-dataset", "ImageDatastore", "Partition", "Test")

# Support multiple datasets (deprecated, kept for compatibility)
DATA_DIRS = [TRAIN_DATA_DIR]
# For backward compatibility
DATA_DIR = DATA_DIRS[0]

# Model checkpoint
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# Image settings
IMAGE_SIZE = 224
NUM_CHANNELS = 3

# Dataset settings
NUM_CLASSES = 12
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Training settings
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# KAN settings
KAN_HIDDEN_DIMS = [64, 32]  # Hidden layers: 512 -> 64 -> 32 -> 12
SPLINE_ORDER = 3  # B-spline order
GRID_SIZE = 5  # Number of grid intervals for splines

# Backbone settings
BACKBONE = "efficientnet_b0"
BACKBONE_FEATURES = 1280  # EfficientNet-B0 output features
FREEZE_BACKBONE = False  # Fine-tune backbone

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed
SEED = 42
