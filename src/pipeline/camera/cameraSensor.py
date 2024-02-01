from common.image.Image import Image
from pipeline.camera.sensorState import SensorState
from abc import ABC, abstractmethod


class CameraSensor(ABC):
    def __init__(
        self, camera_id=0, resolution=(1920, 1080), fps=1, state=SensorState.INIT
    ) -> None:
        """
        function of initiation of a Camera Sensor
        return: None
        """
        self.state = state
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        if state != SensorState.ERROR:
            self.is_active = True
        else:
            self.is_active = False

    @abstractmethod
    def get_img(self) -> Image:
        pass

    @abstractmethod
    def get_state(self) -> SensorState:
        pass
