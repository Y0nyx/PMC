from torch import Tensor
from src.pipeline.feature.featureExtraction import FeatureExtraction
from src.pipeline.data.dataManager import DataManager
from src.common.image.Image import Image


class FeatureExtractionManager:
    _featureExtractions: [FeatureExtraction]
    _dataManager: DataManager

    def __init__(self) -> None:
        """
        Initialize the data manager instance
        :return: None
        """
        self._dataManager = DataManager.get_instance()

    def get_all_features(self) -> [Tensor]:
        """
        Retrieve all features for the different neural network
        :return: array of Tensor
        """
        img: Image = self._dataManager.concate_img()
        return [feature_ex.get_feature(img) for feature_ex in self._featureExtractions]

    def get_feature(self, index_feature_ex: int) -> Tensor:
        """
        Retrieve feature from one neural network
        :return: Tensor
        """
        return self._featureExtractions[index_feature_ex].get_feature(self._dataManager.concate_img())

    def get_state(self) -> []:
        """
        get all the state from the neural network with their name
        :return: list of dict {str : NeuralNetworkState}
        """
        return [{feature_ex.get_name(), feature_ex.get_state()} for feature_ex in self._featureExtractions]
