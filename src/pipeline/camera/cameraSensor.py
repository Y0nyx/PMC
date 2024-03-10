import cv2
import platform
import warnings
from common.image.Image import Image
from pipeline.camera.sensorState import SensorState
from abc import ABC, abstractmethod


class CameraSensor(ABC):
    def __init__(
        self, camera_id=0, standby_resolution=(426, 240), capture_resolution=(1920, 1080), fps=1, verbose: bool = False
    ) -> None:
        """
        function of initiation of a Camera Sensor
        return: None
        """
        self.camera_id = camera_id
        self.standby_resolution = standby_resolution
        self.capture_resolution = capture_resolution
        self.fps = fps

        self.verbose = verbose
        try:
            self.print(f"\nInit camera : {self.camera_id}")
            self.print(f" - standby resolution : {self.standby_resolution}")
            self.print(f" - capture resolution : {self.capture_resolution}")
            self.print(f" - fps : {self.fps}")
            if platform.system() == "Windows":
                cv2.CAP_DSHOW
                # sets the Windows cv2 backend to DSHOW (Direct Video Input Show)
                self.cap = cv2.VideoCapture(self.camera_id)
            elif platform.system() == "Linux":
                cv2.CAP_GSTREAMER  # set the Linux cv2 backend to GTREAMER
                # cv2.CAP_V4L
                self.cap = cv2.VideoCapture(self.camera_id)

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.standby_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.standby_resolution[1])

            # Set frames per second (fps)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

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

    def set_capture_resolution(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_resolution[1])
    
    def set_standby_resolution(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.standby_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.standby_resolution[1])

    def print(self, string):
        if self.verbose:
            print(string)
