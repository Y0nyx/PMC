from common.image.Image import Image
from .webcamCamera import WebcamCamera
from .cameraSensor import CameraSensor
from .sensorState import SensorState

import tqdm
import yaml
from typing import List
from warnings import warn


class CameraManager:
    _instance = None

    def __init__(self, yaml_file, verbose) -> None:
        self.verbose = verbose

        self.print("=== Init CameraManager ===")

        self.yaml_file = yaml_file
        self.cameras = []
        self.read_yaml(yaml_file)
        self.state = SensorState.INIT

    @staticmethod
    def get_instance(yaml_file, verbose: bool = False):
        if CameraManager._instance is None:
            CameraManager._instance = CameraManager(yaml_file, verbose)
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
        with open(yaml_path, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
                camera_configs = yaml_data['cameras']
                with tqdm.tqdm(total=len(camera_configs), desc="cameras init") as pbar:
                    for config in camera_configs:
                        resolution = tuple(map(int, config['resolution'].strip('()').split(','))) or None
                        camera = WebcamCamera(config.get('camera_id', None), resolution, config.get('fps', None), self.verbose)
                        self.add_camera(camera)
                        pbar.update(1)
            except yaml.YAMLError as e:
                warn("Erreur lors de la lecture du fichier YAML : {e}")

    def print(self, string):
        if self.verbose:
            print(string)