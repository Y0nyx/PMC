from pathlib import Path

SAVE_PATH = Path("../../pipeline_results")
SAVE_PATH_UNSUPERVISED_PREDICTION = Path("/unsupervised_prediction")
SAVE_PATH_SEGMENTATION = Path("/segmentation")
SAVE_PATH_SUBDIVISION = Path("/subidivision")
SAVE_PATH_CAPTURE = Path("/capture")

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