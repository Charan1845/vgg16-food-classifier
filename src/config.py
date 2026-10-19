import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Explicit subset of Food-101's 101 classes to train on. data.py remaps each
# selected class's original label to its index in this (sorted) list.
SELECTED_CLASSES = sorted([
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "hamburger", "pizza",
])
NUM_CLASSES = len(SELECTED_CLASSES)

HEAD_DENSE_UNITS = 256
HEAD_DENSE_ACTIVATION = "sigmoid"
DROPOUT_RATE = 0.5

STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3

STAGE2_EPOCHS = 10
STAGE2_LR = 1e-5
FINE_TUNE_BLOCK = "block5"

SEED = 42
