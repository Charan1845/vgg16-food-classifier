import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Food-101 labels are indexed alphabetically 0-100. Keeping only label < NUM_CLASSES
# both subsets the data and keeps label ids contiguous from 0, so no remapping is needed.
# Set to 101 to use the full dataset.
NUM_CLASSES = 15

HEAD_DENSE_UNITS = 256
HEAD_DENSE_ACTIVATION = "sigmoid"
DROPOUT_RATE = 0.5

STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3

STAGE2_EPOCHS = 10
STAGE2_LR = 1e-5
FINE_TUNE_BLOCK = "block5"

SEED = 42
