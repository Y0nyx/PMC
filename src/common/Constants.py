from pathlib import Path

IMG_SAVE_FILE = Path("./dataset/img/")

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

# communication
PIPELINE_PORT = 8080
UNSUPERVISED_PORT = 8081
SUPERVISED_PORT = 8082

PIPELINE_SERVICE = 'pipeline'
UNSUPERVISED_SERVICE = 'unsupervised_learning'
SUPERVISED_SERVICE = 'supervised_learning'