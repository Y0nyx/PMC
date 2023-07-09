from src.common.image.Image import Image
from src.pipeline.camera.cameraSensor import CameraSensor
import cv2
import warnings


class WebcamCamera(CameraSensor):
    def __init__(self, camera_id, state) -> None:
        super().__init__(camera_id, state)

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
            warnings.warn("Erreur : La caméra n'est pas activée.")
            return None
