from common.image.Image import Image
from pipeline.camera.cameraSensor import CameraSensor
from .sensorState import SensorState
import warnings


class WebcamCamera(CameraSensor):
    def __init__(self, camera_id, resolution, fps, verbose: bool = False) -> None:
        super().__init__(camera_id, resolution, fps, verbose)

    def get_img(self) -> Image:
        """
        function to get image from sensor
        :return: Image
        """
        if self.is_active:
            self.print("Capturing image")
            if self.cap.isOpened():
                _ , frame = self.cap.read()
                image = Image(frame)
                return image
        else:
            return None

    def get_state(self) -> SensorState:
        """
        function to get the state of the sensor
        :return: SensorState
        """
        return self.state
