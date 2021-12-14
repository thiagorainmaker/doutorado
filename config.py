import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 80
NUM_EPOCHS = 150
NUM_WORKERS = 6
CHECKPOINT_FILE = "b3.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

