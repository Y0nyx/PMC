from abc import ABC
from torch import Tensor

from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

from common.image.Image import Image
from tools.featureExtraction.featureVector import get_feat_vector
from pipeline.feature.featureExtraction import FeatureExtraction
from common.neuralNetwork.neuralNetworkState import NeuralNetworkState


class FeatureExtractionResnet(FeatureExtraction, ABC):
    def __init__(self) -> None:
        super().__init__()
        self._state = NeuralNetworkState.INIT
        try:
            self._model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self._state = NeuralNetworkState.READY
        except Exception as ex:
            print(ex.args)
            self._state = NeuralNetworkState.ERROR

    def get_feature(self, img: [Image]) -> Tensor:
        try:
            self._state = NeuralNetworkState.PREDICT
            return get_feat_vector(img, self._model)
        except Exception as e:
            print("Resnet feature extraction error")
            print(e.args)
            self._state = NeuralNetworkState.ERROR
            return None

