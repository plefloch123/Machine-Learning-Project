import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "cpu"
DEVICES = 1
PRECISION = 16
