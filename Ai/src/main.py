from pipeline.models.Model import YoloModel
from NetworkManager import NetworkManager
from Pipeline import Pipeline
from common.Constants import *
import time


def path_initialization():

    #Init path to segmentation model
    segmentation_model_path = f"{SEGMENTATION_MODEL_PATH}{SEGMENTATION_MODEL_REF}.pt"
    supervised_detection_model_path = f"{SUPERVISED_DETECTION_MODEL_PATH}{SUPERVISED_DETECTION_MODEL_REF}.pt"

    #Init path to unsupervised detection model
    unsupervised_model_higher_path = ""
    unsupervised_model_path = f'{unsupervised_model_higher_path}{UNSUPERVISED_MODEL_REF}.keras'

    #Init path to current model iteration for logging purposes
    current_iteration_logging_path = f'{SAVE_RESULT}'

    return segmentation_model_path, supervised_detection_model_path, unsupervised_model_path, current_iteration_logging_path



if __name__ == "__main__":

    segmentation_model_path, supervised_detection_model_path, unsupervised_model_path, current_iteration_logging_path  = path_initialization()
    print("ur mom")
    print(segmentation_model_path)
    print(supervised_detection_model_path)
    segmentation_model = YoloModel(Path(segmentation_model_path))
    supervised_detection_model = YoloModel(Path(supervised_detection_model_path))
    print("tabarnak")
    pipeline = Pipeline(segmentation_model=segmentation_model, supervised_detection_model=supervised_detection_model, unsupervised_model=None, current_iteration_logging_path=current_iteration_logging_path)
    print("calisse")
    networkManager = NetworkManager(pipeline, HOST, PORT, SUPERVISED_HOST, SUPERVISED_PORT, UNSUPERVISED_HOST, UNSUPERVISED_PORT, True)
    networkManager.start()


