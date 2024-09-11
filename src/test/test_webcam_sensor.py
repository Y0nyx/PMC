from . import ajuste_import
ajuste_import()

import pytest
from unittest.mock import MagicMock, patch
from common.image import Image
from pipeline.camera import WebcamCamera, SensorState


@pytest.fixture
def webcam_camera():
    with patch("cv2.VideoCapture") as MockVideoCapture:
        mock_cap = MagicMock()
        MockVideoCapture.return_value = mock_cap
        camera = WebcamCamera(camera_id=0, standby_resolution=(426, 240), capture_resolution=(1920, 1080), fps=1, verbose=True)
        camera.cap = mock_cap
        
        with patch.object(camera, 'print', autospec=True) as mock_print:
            camera.print = mock_print
            yield camera

@pytest.mark.short
def test_get_img_exception_handling(webcam_camera):
    webcam_camera.is_active = True
    webcam_camera.cap.read.side_effect = Exception("Mocked exception")

    result = webcam_camera.get_img()

    assert result is None
    assert webcam_camera.get_state() == SensorState.ERROR
    webcam_camera.print.assert_called_with("Error capturing image: Mocked exception")

@pytest.mark.short
def test_get_img_success(webcam_camera):
    webcam_camera.is_active = True
    webcam_camera.cap.isOpened.return_value = True
    webcam_camera.cap.read.return_value = (True, "mock_frame")  
    result = webcam_camera.get_img()

    assert result is not None
    assert isinstance(result, Image)
    webcam_camera.cap.set.assert_called()

@pytest.mark.short
def test_get_img_not_active(webcam_camera):
    webcam_camera.is_active = False

    result = webcam_camera.get_img()

    assert result is None
    assert webcam_camera.get_state() == SensorState.INIT

@pytest.mark.short
def test_get_img_camera_not_opened(webcam_camera):
    webcam_camera.is_active = True
    webcam_camera.cap.isOpened.return_value = False 

    result = webcam_camera.get_img()

    assert result is None
    assert webcam_camera.get_state() == SensorState.ERROR

@pytest.mark.short
def test_get_img_capture_failure(webcam_camera):
    webcam_camera.is_active = True
    webcam_camera.cap.isOpened.return_value = True
    webcam_camera.cap.read.return_value = (False, None) 

    result = webcam_camera.get_img()

    assert result is None
    assert webcam_camera.get_state() == SensorState.ERROR
