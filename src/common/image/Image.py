from dataclasses import dataclass
from PIL import Image as Img
from typing import Union
from pathlib import Path
import numpy as np
import warnings
import cv2


def _validate_img(img: np.ndarray) -> bool:
    """
    Validate image based on color variation
    :param img: Numpy array representing the image
    :return: True if image is valid, False otherwise
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray_img)

    if variance < 450:
        warnings.warn(
            f"Image has very low color variation or is monochromatic. (variance = {variance})"
        )
        return False

    return True


@dataclass
class Image:
    """
    Dataclass of an Image
    """

    _img: np.ndarray = None

    def __init__(self, img: Union[np.ndarray, str, Path]) -> None:
        """
        Init function
        :param img: Either a numpy array representing the image or a file path to load the image from.
        :return: None
        """
        if isinstance(img, Union[str, Path]):
            try:
                img = self.load_img_from_file(img)
            except Exception:
                warnings.warn("Couldn't load image")

        if _validate_img(img):
            self._img = img

    def load_img_from_file(self, file_path: str) -> np.ndarray:
        """
        Load image from file
        :param file_path: Path to the image file
        :return: Numpy array representing the image
        """
        img = Img.open(file_path)
        return np.array(img)

    def __call__(self) -> np.ndarray:
        """
        Get the value of the Image class
        :return:
        """
        return self._img

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
        return self.image.shape

    def resize(self, width, height):
        try:
            cv2.resize(self.image, (width, height))
        except Exception:
            return False
        return True

    def save(self, file_path):
        cv2.imwrite(file_path, self._img)

    def crop(self, boxes):
        image = Img.fromarray(self._img)
        cropped_image = image.crop(boxes.xyxy.tolist()[0])

        return Image(np.array(cropped_image))


if __name__ == "__main__":
    img = Image("D:\\APP\\PMC\\repos\\dataset\\session_5\\photo_camera_0_0.png")

    black_image = np.zeros((100, 100, 3), dtype=np.uint8)

    img = Image(black_image)
