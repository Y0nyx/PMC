from src.common.image.Image import Image
from src.pipeline.camera.sensorState import SensorState
from abc import ABC, abstractmethod


class CameraSensor(ABC):
    def __init__(self) -> None:
        """
        function of initiation of a Camera Sensor
        return: None
        """
        pass

    @abstractmethod
    def get_img(self) -> Image:
        """
        function to get image from sensor
        :return: Image
        """
        pass

    @abstractmethod
    def get_state(self) -> SensorState:
        """
        function to get the state of the sensor
        :return: SensorState
        """
        pass
