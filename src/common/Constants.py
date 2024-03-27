from pathlib import Path

CAPTURED_IMG_SAVE_PATH = Path("./pipeline_results/img/")
SEGMENTED_IMG_SAVE_PATH = Path("./pipeline_results/capture/")
UNSUPERVISED_PREDICTED_SAVE_PATH = Path("./pipeline_results/unsupervised_prediction")
UNSUPERVISED_PREDICTED_SAVE_PATH_SEGMENTATION = Path("/segmentation")
UNSUPERVISED_PREDICTED_SAVE_PATH_SUBDIVISION = Path("/subidivision")

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