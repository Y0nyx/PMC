from abc import ABC

from torch import Tensor

from src.common.image.Image import Image
from src.pipeline.feature.featureExtraction import FeatureExtraction


class FeatureExtractionResnet(FeatureExtraction, ABC):
    def __init__(self):
        super().__init__()

    def get_feature(self, img: Image) -> Tensor:
        pass

