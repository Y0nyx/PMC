import unittest
from unittest.mock import MagicMock, patch, mock_open

import os
import sys
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.image.ImageCollection import ImageCollection
from common.image.Image import Image
from pipeline.camera.cameraSensor import CameraSensor
from pipeline.camera.sensorState import SensorState
from pipeline.camera.cameraManager import CameraManager


class TestCameraManager(unittest.TestCase):

    def setUp(self):
        # Mock dependencies
        self.yaml_content = """
        cameras:
          - camera_id: 1
            capture_resolution: "(1920, 1080)"
            standby_resolution: "(640, 480)"
            fps: 30
        """
        self.mock_yaml_file = "mock_config.yaml"
        self.verbose = False

        # Create instance of CameraManager with mocked YAML file
        with patch("builtins.open", mock_open(read_data=self.yaml_content)):
            self.manager = CameraManager(self.mock_yaml_file, self.verbose)

    def test_singleton_instance(self):
        with open(self.mock_yaml_file, 'w') as file:
            file.write(self.yaml_content)
            file.close()
        # Test if singleton pattern works correctly
        manager1 = CameraManager.get_instance(self.mock_yaml_file, self.verbose)
        manager2 = CameraManager.get_instance(self.mock_yaml_file, self.verbose)
        self.assertIs(manager1, manager2)

        os.remove(self.mock_yaml_file)

    def test_add_camera(self):
        # Test add_camera method
        mock_camera = MagicMock(spec=CameraSensor)
        result = self.manager.add_camera(mock_camera)
        self.assertTrue(result)
        self.assertIn(mock_camera, self.manager.cameras)

    def test_add_invalid_camera(self):
        # Test adding an invalid camera
        invalid_camera = object()  # Not a CameraSensor
        result = self.manager.add_camera(invalid_camera)
        self.assertFalse(result)
        self.assertNotIn(invalid_camera, self.manager.cameras)

    def test_remove_camera(self):
        # Test removing a camera by index
        mock_camera = MagicMock(spec=CameraSensor)
        self.manager.cameras = []
        self.manager.add_camera(mock_camera)
        result = self.manager.remove_camera(0)
        self.assertTrue(result)
        self.assertNotIn(mock_camera, self.manager.cameras)

    def test_remove_invalid_camera(self):
        # Test removing a camera with an invalid index
        result = self.manager.remove_camera(10)
        self.assertFalse(result)

    def test_get_all_img(self):
        # Test get_all_img method
        mock_camera = MagicMock(spec=CameraSensor)
        mock_image = MagicMock(spec=Image)
        mock_camera.get_img.return_value = mock_image
        self.manager.cameras = []
        self.manager.add_camera(mock_camera)
        self.manager.add_camera(mock_camera)
        
        images = self.manager.get_all_img()
        self.assertIsInstance(images, ImageCollection)
        self.assertEqual(images.img_count, 2)

    def test_get_img(self):
        # Test get_img method
        mock_camera = MagicMock(spec=CameraSensor)
        mock_image = MagicMock(spec=Image)
        mock_camera.get_img.return_value = mock_image
        self.manager.add_camera(mock_camera)
        
        image = self.manager.get_img(0)
        self.assertIsInstance(image, Image)

    def test_get_img_invalid_index(self):
        # Test get_img with an invalid index
        image = self.manager.get_img(10)
        self.assertIsNone(image)

    def test_get_state(self):
        # Test get_state method
        mock_camera = MagicMock(spec=CameraSensor)
        mock_camera.get_state.return_value = SensorState.READY
        self.manager.add_camera(mock_camera)
        
        state = self.manager.get_state()
        self.assertEqual(state, SensorState.READY)

    def test_get_state_with_error(self):
        # Test get_state when a camera reports an error
        mock_camera = MagicMock(spec=CameraSensor)
        mock_camera.get_state.return_value = SensorState.ERROR
        self.manager.add_camera(mock_camera)
        
        state = self.manager.get_state()
        self.assertEqual(state, SensorState.ERROR)

    @patch("tqdm.tqdm")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    @patch("yaml.safe_load")
    def test_read_yaml(self, mock_safe_load, mock_open, mock_tqdm):
        # Test read_yaml method with valid data
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
        self.manager.cameras = []
        self.manager.read_yaml(self.mock_yaml_file)
        self.assertEqual(len(self.manager.cameras), 1)

    @patch("builtins.open", new_callable=mock_open)
    def test_read_yaml_invalid(self, mock_open):
        # Test read_yaml method with invalid data
        mock_open.side_effect = yaml.YAMLError("Mocked YAML Error")
        with self.assertRaises(yaml.YAMLError):
            self.manager.read_yaml(self.mock_yaml_file)

    def test_print(self):
        # Test print method
        with patch("builtins.print") as mock_print:
            self.manager.print("Test message")
            if self.verbose:
                mock_print.assert_called_once_with("Test message")
            else:
                mock_print.assert_not_called()

if __name__ == "__main__":
    unittest.main()
