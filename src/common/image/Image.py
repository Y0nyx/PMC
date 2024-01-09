from dataclasses import dataclass
import numpy as np
import cv2


def _validate_img(img: np.ndarray) -> bool:
    return True


@dataclass
class Image:
    """
    Dataclass of an Image
    """
    _img: np.ndarray = None

    def __init__(self, img: np.ndarray) -> None:
        """
        Init function
        :param img:
        return None
        """
        if _validate_img(img):
            self._img = img
        else:
            raise Exception("Not valid image")

    @property
    def value(self) -> np.ndarray:
        """
        Get the value of the Image class
        :return:
        """
        return self._img

    @value.setter
    def value(self, img: np.ndarray) -> None:
        """
        Set the value of the Image class
        :param img:
        :return:
        """
        if _validate_img(img):
            self._img = img
        else:
            raise Exception("Not valid image")

    def get_size(self):
        height, width, channels = self.image.shape
        return channels, height, width

    def resize(self, width, height):
        cv2.resize(self.image, (width, height))
        return True
