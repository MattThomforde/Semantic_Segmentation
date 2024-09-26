import torch
import os

# Define paths for images and masks
TRAIN_IMG_DIR = 'data/train_images/'
TRAIN_MASK_DIR = 'data/train_masks/'
VAL_IMG_DIR = 'data/validation_images/'
VAL_MASK_DIR = 'data/validation_masks/'

# Determine the device to be used
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Determine whether to pin memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# Set learning rate, number of epochs to train for, and batch size
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 32

# Define the input image dimensions to scale to
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 192

# Use an existing model?
LOAD_MODEL = False

# Define threshold for filtering predictions
THRESHOLD = 0.18

# Define the path to the base output directory
BASE_OUTPUT = "output"

# Define the path to the saved model
MODEL_PATH = os.path.join(BASE_OUTPUT, "my_checkpoint.pth")

