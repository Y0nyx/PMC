from pipeline.models.Model import YoloModel
from NetworkManager import NetworkManager
from Pipeline import Pipeline
from common.Constants import *
import tensorflow as tf


def path_initialization():

    #Init path to segmentation model
    segmentation_model_path = f"{SEGMENTATION_MODEL_PATH}{SEGMENTATION_MODEL_REF}.pt"

    #Init path to unsupervised detection model
    unsupervised_model_higher_path = ""
    unsupervised_model_path = f'{unsupervised_model_higher_path}{UNSUPERVISED_MODEL_REF}.keras'

    #Init path to current model iteration for logging purposes
    current_iteration_logging_path = f'{SAVE_RESULT}'

    return segmentation_model_path, unsupervised_model_path, current_iteration_logging_path



if __name__ == "__main__":

    segmentation_model_path, unsupervised_model_path, current_iteration_logging_path  = path_initialization()

    segmentation_model = YoloModel(Path(segmentation_model_path))

    pipeline = Pipeline(segmentation_model=segmentation_model, supervised_detection_model=None, unsupervised_model=None, current_iteration_logging_path=current_iteration_logging_path)
    networkManager = NetworkManager(pipeline, HOST, PORT, SUPERVISED_HOST, SUPERVISED_PORT, UNSUPERVISED_HOST, UNSUPERVISED_PORT, True)
    networkManager.start()
