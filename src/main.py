from pipeline.models.Model import YoloModel
from NetworkManager import NetworkManager
from Pipeline import Pipeline
from common.Constants import *


if __name__ == '__main__':
    supervised_models = [YoloModel(Path("./ia/segmentation/v1.pt"))]
    pipeline = Pipeline(supervised_models=supervised_models, unsupervised_models=[])

    networkManager = NetworkManager(pipeline, HOST, PORT, True)
    networkManager.start()