import pathlib
import yaml
import numpy as np
from src.pipeline.camera.cameraManager import CameraManager #May have to be modified depending on implementation
from src.common.image.Image import Image

class DataManager:

    def __init__(self, yaml_path: Path) -> None:
        """
        Initializes the DataManager.
        """
        self._yaml_path = yaml_path
        self._param = _read_yaml()
        self._camera_manager = CameraManager.get_instance();


    def concate_img(self) -> Image:
        """
        Concatenates the images from the CameraManager into one Image object.
        :param self:
        :Returns:
            Image: Image object containing the value attribute of all images concatenated as a one dimension NumPy array.
        """
        
        #Get all the images from the CameraManager
        images = self._camera_manager.get_all_img()

        #Create a null ndArray that will contain all the image values to be concatenated
        concatenated_image_values = np.empty()

        # TODO: Implement different concatenating algorithms to fit the self._param attribute

        # Iterate over the list of images
        for img in images:
            concatenated_image_values = np.concatenate((concatenated_image_values, img.value), axis=1)

        #Create the Image object containing the concatenated values
        concatenated_images = Image(concatenated_image_values)

        return concatenated_images


    def _read_yaml(self) -> None:
        """
        Function to read yaml
        :param self: 
        :return: None
        """

        with open(self.yaml_path, 'r') as file:
            try:
                self.param = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        

