from . import ajuste_import
ajuste_import()

import os
import yaml
import pytest
from unittest.mock import MagicMock, patch, mock_open
from common.image import ImageCollection, Image
from pipeline.camera import CameraSensor, SensorState, CameraManager

@pytest.fixture
def camera_manager():
    yaml_content = """
    cameras:
      - camera_id: 1
        capture_resolution: "(1920, 1080)"
        standby_resolution: "(640, 480)"
        fps: 30
    """
    mock_yaml_file = "mock_config.yaml"
    verbose = False
    
    with open(mock_yaml_file, 'w')  as file:
        file.write(yaml_content)
        file.close()

    # Create instance of CameraManager with mocked YAML file
    manager = CameraManager(mock_yaml_file, verbose)

    yield manager
    # Cleanup
    if os.path.exists(mock_yaml_file):
        os.remove(mock_yaml_file)

@pytest.mark.long
def test_singleton_instance():
    yaml_content = """
    cameras:
      - camera_id: 1
        capture_resolution: "(1920, 1080)"
        standby_resolution: "(640, 480)"
        fps: 30
    """
    mock_yaml_file = "mock_config.yaml"
    with open(mock_yaml_file, 'w') as file:
        file.write(yaml_content)

    manager1 = CameraManager.get_instance(mock_yaml_file)
    manager2 = CameraManager.get_instance(mock_yaml_file)
    
    assert manager1 is manager2

    os.remove(mock_yaml_file)

@pytest.mark.long
def test_add_camera(camera_manager):
    mock_camera = MagicMock(spec=CameraSensor)
    result = camera_manager.add_camera(mock_camera)
    assert result
    assert mock_camera in camera_manager.cameras

@pytest.mark.long
def test_add_invalid_camera(camera_manager):
    invalid_camera = object()
    with pytest.warns(UserWarning, match="Erreur : L'objet .* n'est pas une instance de CameraSensor."):
        result = camera_manager.add_camera(invalid_camera)
    assert not result
    assert invalid_camera not in camera_manager.cameras

@pytest.mark.long
def test_remove_camera(camera_manager):
    mock_camera = MagicMock(spec=CameraSensor)
    camera_manager.cameras = []
    camera_manager.add_camera(mock_camera)
    result = camera_manager.remove_camera(0)
    assert result
    assert mock_camera not in camera_manager.cameras

@pytest.mark.long
def test_remove_invalid_camera(camera_manager):
    with pytest.warns(UserWarning, match="Erreur : Index de caméra invalide."):
        result = camera_manager.remove_camera(10)
    assert not result

@pytest.mark.long
def test_get_all_img(camera_manager):
    mock_camera = MagicMock(spec=CameraSensor)
    mock_image = MagicMock(spec=Image)
    mock_camera.get_img.return_value = mock_image
    camera_manager.cameras = []
    camera_manager.add_camera(mock_camera)
    camera_manager.add_camera(mock_camera)
    
    images = camera_manager.get_all_img()
    assert isinstance(images, ImageCollection)
    assert images.img_count == 2

@pytest.mark.long
def test_get_img(camera_manager):
    mock_camera = MagicMock(spec=CameraSensor)
    mock_image = MagicMock(spec=Image)
    mock_camera.get_img.return_value = mock_image
    camera_manager.cameras = []
    camera_manager.add_camera(mock_camera)
    
    image = camera_manager.get_img(0)
    assert isinstance(image, Image)

@pytest.mark.long
def test_get_img_invalid_index(camera_manager):
    with pytest.warns(UserWarning, match="Erreur : Index de caméra invalide."):
        image = camera_manager.get_img(10)
    assert image is None

@pytest.mark.long
def test_get_state(camera_manager):
    mock_camera = MagicMock(spec=CameraSensor)
    mock_camera.get_state.return_value = SensorState.READY
    camera_manager.add_camera(mock_camera)
    
    state = camera_manager.get_state()
    assert state == SensorState.READY

@pytest.mark.long
def test_get_state_with_error(camera_manager):
    mock_camera = MagicMock(spec=CameraSensor)
    mock_camera.get_state.return_value = SensorState.ERROR
    camera_manager.add_camera(mock_camera)
    
    state = camera_manager.get_state()
    assert state == SensorState.ERROR

@pytest.mark.long
@patch("tqdm.tqdm")
@patch("builtins.open", new_callable=mock_open, read_data="")
@patch("yaml.safe_load")
def test_read_yaml(mock_safe_load, mock_open, mock_tqdm, camera_manager):
    mock_safe_load.return_value = {
        "cameras": [
            {
                "camera_id": 1,
                "capture_resolution": "(1920, 1080)",
                "standby_resolution": "(640, 480)",
                "fps": 30
            }
        ]
    }
    camera_manager.cameras = []
    camera_manager.read_yaml(camera_manager.yaml_file)
    assert len(camera_manager.cameras) == 1

@pytest.mark.long
@patch("builtins.open", new_callable=mock_open)
def test_read_yaml_invalid(mock_open, camera_manager):
    mock_open.side_effect = yaml.YAMLError("Mocked YAML Error")
    with pytest.raises(yaml.YAMLError):
        camera_manager.read_yaml(camera_manager.yaml_file)

@pytest.mark.long
def test_print(camera_manager):
    with patch("builtins.print") as mock_print:
        camera_manager.verbose = True
        camera_manager.print("Test message")
        mock_print.assert_called_once_with("Test message")

        mock_print.reset_mock()
        
        camera_manager.verbose = False
        camera_manager.print("Test message")
        mock_print.assert_not_called()

@pytest.mark.long
def test_reset_instance(camera_manager):
    camera_manager.reset_instance()
    assert camera_manager._instance == None
