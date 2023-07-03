import yaml
from torch import Tensor
from pathlib import Path
from abc import ABC, abstractmethod

from src.common.image.Image import Image
from src.common.neuralNetwork.neuralNetworkState import NeuralNetworkState


class FeatureExtraction(ABC):
    _state: NeuralNetworkState
    _yaml_path: Path
    _param: {}
    _name: str

    def __init__(self) -> None:
        """
        function of initiation of a Feature Extraction
        return: None
        """
        pass

    @abstractmethod
    def get_feature(self, img: Image) -> Tensor:
        """
        get feature from the neural network
        :return: Tensor
        """
        pass

    def get_state(self) -> NeuralNetworkState:
        return self._state

    def get_name(self) -> str:
        """
        get the name of the neural network
        :return: str
        """
        return self._name

    def _read_yaml(self) -> None:
        """
        Function to read yaml
        :return: None
        """
        with open(self._yaml_path, 'r') as file:
            try:
                self._param = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
