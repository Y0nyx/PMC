import cv2
import warnings
from common.image.Image import Image
from pipeline.camera.sensorState import SensorState
from abc import ABC, abstractmethod


class CameraSensor(ABC):
    def __init__(
        self, camera_id=0, resolution=(1920, 1080), fps=1, verbose: bool = False
    ) -> None:
        """
        function of initiation of a Camera Sensor
        return: None
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps

        self.verbose = verbose
        try:
            self.print(f"Init camera : {self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id) 

            # Set resolution
            #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Set frames per second (fps)
            #self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            self.is_active = True
            self.state = SensorState.INIT
            self.print(f"Finish Init camera : {self.camera_id}")
        except Exception:
            warnings.warn("Erreur : La caméra n'est pas activée.")
            self.is_active = False
            self.state = SensorState.ERROR

    @abstractmethod
    def get_img(self) -> Image:
        pass

    @abstractmethod
    def get_state(self) -> SensorState:
        pass

    def print(self, string):
        if self.verbose:
            print(string)
