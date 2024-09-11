from pipeline.models.Model import YoloModel
from NetworkManager import NetworkManager
from Pipeline import Pipeline
from common.constants import *
import tensorflow as tf

def count_folders_starting_with(start_string, path):
    count = 0
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.startswith(start_string):
                count += 1
    return count

def path_initialization():

    #Init path to segmentation model
    segmentation_model_path = f"{SEGMENTATION_MODEL_PATH}{SEGMENTATION_MODEL_REF}.pt"

    #Init path to unsupervised detection model
    unsupervised_model_higher_path = ""
    unsupervised_model_path = f'{unsupervised_model_higher_path}{UNSUPERVISED_MODEL_REF}.keras'

    #Init path to current model iteration for logging purposes
    current_iteration_latest_version_id = count_folders_starting_with(f"{UNSUPERVISED_MODEL_REF}", unsupervised_model_higher_path)
    current_iteration_logging_path = f'/{UNSUPERVISED_MODEL_REF}_{current_iteration_latest_version_id}'

    return segmentation_model_path, unsupervised_model_path, current_iteration_logging_path



if __name__ == "__main__":

    segmentation_model_path, unsupervised_model_path, current_iteration_logging_path  = path_initialization()

    supervised_models = [YoloModel(Path(segmentation_model_path))]
    unsupervised_model = tf.keras.models.load_model(unsupervised_model_path)

    pipeline = Pipeline(supervised_models=supervised_models, unsupervised_model=unsupervised_model, current_iteration_logging_path=current_iteration_logging_path)
    networkManager = NetworkManager(pipeline, HOST, PORT, SUPERVISED_HOST, SUPERVISED_PORT, UNSUPERVISED_HOST, UNSUPERVISED_PORT, True)
    networkManager.start()
