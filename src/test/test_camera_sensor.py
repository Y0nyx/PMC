from . import ajuste_import
ajuste_import()

import pytest
import cv2
from unittest.mock import patch, MagicMock
from pipeline.camera import CameraSensor, SensorState


class _TestCameraSensor(CameraSensor):
    def __init__(self, camera_id=0, standby_resolution=(426, 240), capture_resolution=(1920, 1080), fps=1, verbose=False):
        super().__init__(camera_id, standby_resolution, capture_resolution, fps, verbose)
    
    def get_img(self):
        return MagicMock()
    
    def get_state(self):
        return self.state


@pytest.fixture
def mock_videocapture():
    with patch('cv2.VideoCapture') as mock_capture:
        yield mock_capture


@pytest.fixture
def mock_platform():
    with patch('platform.system') as mock_system:
        yield mock_system


@pytest.fixture
def camera_sensor(mock_videocapture, mock_platform):
    mock_platform.return_value = 'Windows'
    return _TestCameraSensor(camera_id=1, verbose=True)


def test_camera_sensor_init_windows(mock_videocapture, mock_platform):
    mock_platform.return_value = 'Windows'
    mock_cap = mock_videocapture.return_value

    sensor = _TestCameraSensor(camera_id=1, verbose=True)

    assert sensor.camera_id == 1
    assert sensor.standby_resolution == (426, 240)
    assert sensor.capture_resolution == (1920, 1080)
    assert sensor.fps == 1
    assert sensor.is_active is True
    assert sensor.state == SensorState.INIT
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 426)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 1)


def test_camera_sensor_init_linux(mock_videocapture, mock_platform):
    mock_platform.return_value = 'Linux'
    mock_cap = mock_videocapture.return_value

    sensor = _TestCameraSensor(camera_id=2, verbose=False)

    assert sensor.camera_id == 2
    assert sensor.standby_resolution == (426, 240)
    assert sensor.capture_resolution == (1920, 1080)
    assert sensor.fps == 1
    assert sensor.is_active is True
    assert sensor.state == SensorState.INIT
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 426)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 1)


def test_camera_sensor_init_error(mock_videocapture):
    mock_videocapture.side_effect = Exception("Camera error")

    with pytest.warns(UserWarning, match="Erreur : La caméra n'est pas activée."):
        sensor = _TestCameraSensor(camera_id=3)

    assert sensor.is_active is False
    assert sensor.state == SensorState.ERROR


def test_set_capture_resolution(camera_sensor):
    mock_cap = camera_sensor.cap
    camera_sensor.set_capture_resolution()
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def test_set_standby_resolution(camera_sensor):
    mock_cap = camera_sensor.cap
    camera_sensor.set_standby_resolution()
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 426)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 240)


def test_print(camera_sensor, capsys):
    camera_sensor.print("Test message")
    captured = capsys.readouterr()
    assert "Test message" in captured.out
