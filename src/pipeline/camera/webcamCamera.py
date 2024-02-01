from common.image.Image import Image
from pipeline.camera.cameraSensor import CameraSensor
from .sensorState import SensorState
import cv2
import warnings


class WebcamCamera(CameraSensor):
    def __init__(self, camera_id, resolution, fps) -> None:
        super().__init__(camera_id, resolution, fps, SensorState.INIT)

    def get_img(self) -> Image:
        """
        function to get image from sensor
        :return: Image
        """
        if self.is_active:
            print("Capturing image")
            cap = cv2.VideoCapture(self.camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                cv2.imwrite('./test.png', frame)
                image = Image(frame)
                # ------------------------------------------------
                # To be removed, mais je veux voir ma sale tete quand je debug
                # cv2.imshow('Captured Image', frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # ------------------------------------------------
                return image
        else:
            warnings.warn("Erreur : La caméra n'est pas activée.")
            return None

    def get_state(self) -> SensorState:
        """
        function to get the state of the sensor
        :return: SensorState
        """
        return self.state.value
