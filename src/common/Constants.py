from pathlib import Path

IMG_SAVE_FILE = Path("./dataset/img/")
IMG_CAPTURE_FILE = Path("./dataset/capture/")

# training loop
NUMBER_IMG_THRESHOLD = 10
TIME_THRESHOLD = 21
TRAIN_SPLIT = 0.8

TRAIN_FILE = Path("./dataset/train/")
VALID_FILE = Path("./dataset/valid/")
YAML_FILE = Path('./dataset/data.yaml')

EPOCHS = 3
BATCH = -1
NC = 1
CLASSES = ['default']

# Connection Electron
PORT = 8002
HOST = 'host.docker.internal'