from common.image.Image import Image
from pipeline.camera.cameraSensor import CameraSensor
from .sensorState import SensorState

class WebcamCamera(CameraSensor):
    def __init__(self, camera_id, standby_resolution, capture_resolution, fps, verbose: bool = False) -> None:
        super().__init__(camera_id, standby_resolution, capture_resolution, fps, verbose)

    def get_img(self) -> Image:
        """
        function to get image from sensor
        :return: Image
        """
        if self.is_active:
            try:
                self.print("Setting capture resolution")
                self.set_capture_resolution()
                self.print("Capturing image")
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.print("Failed to capture image")
                        self.state = SensorState.ERROR
                        return None
                    image = Image(frame)
                    self.print("Camera back in standby mode")
                    self.set_standby_resolution()
                    return image
                else:
                    self.print("Camera is not opened")
                    self.state = SensorState.ERROR
                    return None
            except Exception as e:
                self.print(f"Error capturing image: {str(e)}")
                self.state = SensorState.ERROR
                return None
        else:
            self.print("Camera is not active, back in standby mode")
            return None


    def get_state(self) -> SensorState:
        """
        function to get the state of the sensor
        :return: SensorState
        """
        return self.state
