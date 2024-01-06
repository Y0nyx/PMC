from src.common.image.Image import Image
from src.pipeline.camera.sensorState import SensorState
from abc import ABC, abstractmethod


class CameraSensor(ABC):
    def __init__(self, camera_id, state) -> None:
        """
        function of initiation of a Camera Sensor
        return: None
        """
        self.state = state
        self.camera_id = camera_id
        self.is_active = False

    @abstractmethod
    def get_img(self) -> Image:
        pass

    @abstractmethod
    def get_state(self) -> SensorState:
        """
        function to get the state of the sensor
        :return: SensorState
        """
        return self.state.value
