from common.image.Image import Image
from pipeline.camera.cameraSensor import CameraSensor
from .sensorState import SensorState
import warnings


class WebcamCamera(CameraSensor):
    def __init__(self, camera_id, standby_resolution, capture_resolution, fps, verbose: bool = False) -> None:
        super().__init__(camera_id, standby_resolution, capture_resolution, fps, verbose)

    def get_img(self) -> Image:
        """
        function to get image from sensor
        :return: Image
        """
        if self.is_active:
            self.print("Setting capture resolution")
            self.set_capture_resolution()
            self.print("Capturing image")
            if self.cap.isOpened():
                _, frame = self.cap.read()
                image = Image(frame)
                self.print("Setting standby resolution")
                self.set_standby_resolution()
                return image
        else:
            self.print("Setting standby resolution")
            return None

    def get_state(self) -> SensorState:
        """
        function to get the state of the sensor
        :return: SensorState
        """
        return self.state
