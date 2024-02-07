from common.image.Image import Image
from .webcamCamera import WebcamCamera
from .cameraSensor import CameraSensor
from .sensorState import SensorState

import os
import yaml
from typing import List
from warnings import warn


class CameraManager:
    _instance = None

    def __init__(self, yaml_file) -> None:
        self.yaml_file = yaml_file
        self.nbr_camera = 1
        self.cameras = []
        self.read_yaml(yaml_file)
        self.state = SensorState.INIT

    @staticmethod
    def get_instance(yaml_file):
        if CameraManager._instance is None:
            CameraManager._instance = CameraManager(yaml_file)
        return CameraManager._instance

    def add_camera(self, *cameras):
        for camera in cameras:
            if isinstance(camera, CameraSensor):
                self.cameras.append(camera)
                return True
            else:
                warn("Erreur : L'objet {camera} n'est pas une instance de CameraSensor.")
                return False

    def remove_camera(self, index_camera):
        if index_camera < len(self.cameras):
            del self.cameras[index_camera]
            return True
        else:
            warn("Erreur : Index de caméra invalide.")
            return False

    def get_all_img(self) -> List[Image]:
        """
        function to get all the image of all camera
        :return: image
        """
        images = []
        for camera in self.cameras:
            image = camera.get_img()
            images.append(image)
        return images

    def get_img(self, index_camera) -> Image:
        """
        function to gat the image of a camera
        :return: image
        """
        if 0 <= index_camera < len(self.cameras):
            camera = self.cameras[index_camera]
            return camera.get_img()
        else:
            warn("Erreur : Index de caméra invalide.")
            return None

    def get_state(self) -> SensorState:
        for camera in self.cameras:
            if camera.get_state() == SensorState.ERROR:
                self.state = SensorState.ERROR
                break
        else:
            self.state = SensorState.READY
        return self.state

    def read_yaml(self, yaml_path):
        print(os.getcwd())
        with open(yaml_path, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
                camera_configs = yaml_data['cameras']
                for config in camera_configs:
                    camera = WebcamCamera(config['camera_id'], config['resolution'], config['fps'])
                    self.add_camera(camera)
            except yaml.YAMLError as e:
                warn("Erreur lors de la lecture du fichier YAML : {e}")
