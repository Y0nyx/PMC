from pathlib import Path
import os

SAVE_PATH = Path("")
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
# batch -1 : auto-batch size
BATCH = -1
NC = 1
CLASSES = ['default']

# Connection Electron
PORT = 8002
HOST = '127.0.0.1'
SUPERVISED_PORT = 8003
SUPERVISED_HOST = '127.0.0.1'
UNSUPERVISED_PORT = 8004
UNSUPERVISED_HOST = '127.0.0.1'

#TODO: Ajouter les constantes pour le CSV
CSV_FIELDS = ['id', 'un_sup_model_ref', 'seg_model_ref', 'sup_model_ref',
                           'capture_img_path', 'seg_results', 'seg_threshold', 'seg_img_path',
                           'sub_img_path', 'unsup_pred_img_path', 'unsup_defect_threshold',
                           'unsup_threshold_algo', 'unsup_defect_res', 'unsup_defect_bb',
                           'sup_defect_res', 'sup_defect_threshold', 'sup_defect_bb',
                           'manual_verification_result', 'datetime']

CSV_FILE_NAME = SAVE_PATH / "results.csv"

UNSUPERVISED_MODEL_REF = "simple_cae_blackout_model"
SEGMENTATION_MODEL_REF = "segmentation_v3"
SUPERVISED_DETECTION_MODEL_REF = "supervised_detection"

SEGMENTATION_MODEL_PATH = "./ai_models/segmentation/"
SUPERVISED_DETECTION_MODEL_PATH = "./ai_models/detection/"
SAVE_RESULT = Path("/pipeline_analysis/")