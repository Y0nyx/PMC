from pathlib import Path
import numpy as np
import yaml
import cv2
from common.image.Image import Image
from ..camera.cameraManager import CameraManager #May have to be modified depending on implementation
from ..camera.webcamCamera import WebcamCamera
from ..camera.sensorState import SensorState

class DataManager:
    instance = None

    def __init__(self, yaml_path: Path = "") -> None:
        """
        Initializes the DataManager.
        """
        self._yaml_path = yaml_path
        self._param = self._read_yaml()
        self._camera_manager = CameraManager.get_instance("")
        webcamCamera = WebcamCamera(0, SensorState.INIT)
        self._camera_manager.add_camera(webcamCamera)

    @staticmethod
    def get_instance():
        if DataManager.instance is None:
            DataManager.instance = DataManager()
        print("Hello get_instance")
        return DataManager.instance

    def concate_img(self) -> [Image]:
        """
        Concatenates the images from the CameraManager into one Image object.
        :param self:
        :Returns:
            Image: Image object containing the value attribute of all images concatenated as a one dimension NumPy array.
        """
        #Get all the images from the CameraManager and preprocess them
        images = self._apply_preprocessing(self._camera_manager.get_all_img())

        #Show images after preprocessing
        for image in images:
            cv2.imshow('Captured Image', image.value)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #Create a null ndArray that will contain all the image values to be concatenated
        concatenated_image_values = np.empty(0)

        # TODO: Implement different concatenating algorithms to fit the self._param attribute

        # Iterate over the list of images
        for img in images:
            concatenated_image_values = np.concatenate((concatenated_image_values, img.value.flatten()), axis=0)

        #Create the Image object containing the concatenated values
        concatenated_images = Image(concatenated_image_values)

        return images

    def _apply_preprocessing(self, images: [Image])->[Image]:
        """
        Apply preprocessing algorithms as specified in the yaml params
        :params:
            self
            images: List of images to preprocess
        :Returns:
            images: Preprocessed images
        """
        #if(condition):
        #    images = _grayscale_conversion(images)
        #if(condition):
        #    images = _normalization_conversion(images)
        #if(condition):
        #    images = _equalized_conversion(images)
        #if(condition):
        #    images = _gaussian_blur_conversion(images)
        #if(condition):
        #    images = _threshold_conversion(images)
        #if(condition):
        #    images = _edge_detection_conversion(images)
        return images

    def _grayscale_conversion(self, images: [Image])->[Image]:
        """
        Apply grayscale conversion
        :params:
            self
            images: List of images to convert
        :Returns:
            images: Converted images
        """
        for image in images:
            image.value = cv2.cvtColor(image.value, cv2.COLOR_BGR2GRAY)

        return images

    def _normalization_conversion(self, images: [Image]) -> [Image]:
        """
        Apply normalization to images.
        :param self:
        :param images: List of images to normalize.
        :return: Normalized images.
        """
        for image in images:
            image.value = image.value / 255.0

        return images

    def _equalized_conversion(self, images: [Image]) -> [Image]:
        """
        Apply histogram equalization to images after converting them to grayscale.
        :param self:
        :param images: List of images to equalize.
        :return: Equalized images.
        """
        images = self._grayscale_conversion(images)
        for image in images:
            image.value = cv2.equalizeHist(image.value)

        return images

    def _gaussian_blur_conversion(self, images: [Image]) -> [Image]:
        """
        Apply Gaussian blur to images.
        :param self:
        :param images: List of images to apply Gaussian blur.
        :return: Blurred images.
        """
        for image in images:
            image.value = cv2.GaussianBlur(image.value, (5, 5), 0)

        return images

    def _threshold_conversion(self, images: [Image]) -> [Image]:
        """
        Apply thresholding to images.
        :param self:
        :param images: List of images to apply thresholding.
        :return: Thresholded images.
        """
        for image in images:
            _, image.value = cv2.threshold(image.value, 127, 255, cv2.THRESH_BINARY)

        return images

    def _edge_detection_conversion(self, images: [Image]) -> [Image]:
        """
        Apply edge detection to images.
        :param self:
        :param images: List of images to apply edge detection.
        :return: Images with edge detection applied.
        """
        for image in images:
            image.value = cv2.Canny(image.value, 100, 200)

        return images


    def _read_yaml(self) -> None:
        """
        Function to read yaml
        :param self: 
        :return: None
        """

        # with open(self.yaml_path, 'r') as file:
        #     try:
        #         self.param = yaml.safe_load(file)
        #     except yaml.YAMLError as exc:
        #         print(exc)
        

