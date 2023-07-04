from src.common.image.Image import Image
from src.pipeline.camera.sensorState import SensorState
from abc import ABC
import cv2


class CameraSensor(ABC):
    def __init__(self, camera_id, state) -> None:
        """
        function of initiation of a Camera Sensor
        return: None
        """
        self.state = state
        self.camera_id = camera_id
        self.is_active = False

    def get_img(self) -> Image:
        """
        function to get image from sensor
        :return: Image
        """
        if self.is_active:
            cap = cv2.VideoCapture(self.camera_id)
            ret, frame = cap.read()
            cap.release()
            return frame
        else:
            print("Erreur : La caméra n'est pas activée.")
            return None

    def get_state(self) -> SensorState:
        """
        function to get the state of the sensor
        :return: SensorState
        """
        return self.state.value
