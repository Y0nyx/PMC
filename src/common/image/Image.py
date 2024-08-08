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
    # TODO Rajouter différentes erreurs de cam possibles
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

        # if _validate_img(img):
        self._img = img
        self._mask = None

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

    @property
    def mask(self) -> np.ndarray:
        """
        Get the mask of the Image class.
        :return: Numpy array representing the mask.
        """
        return self._mask

    @mask.setter
    def mask(self, mask: np.ndarray) -> None:
        """
        Set the mask of the Image class.
        :param mask: Numpy array representing the mask.
        :return: None
        """
        self._mask = mask

    @property
    def shape(self):
        """
        Get the shape of the Image class
        :return:
        """
        return self._img.shape

    def resize(self, width, height) -> bool:
        """
        resize image
        :return: bool
        """
        try:
            cv2.resize(self.image, (width, height))
        except Exception:
            return False
        return True

    def save(self, file_path) -> None:
        """
        save image to path, must include name + extension ex: img.png
        :return:
        """
        cv2.imwrite(file_path, self._img)

    def crop(self, boxes):
        image = Img.fromarray(self._img)
        cropped_image = image.crop(boxes.xyxy.tolist()[0])

        mask = Img.fromarray(self._mask)
        cropped_mask = mask.crop(boxes.xyxy.tolist()[0])

        image_obj = Image(np.array(cropped_image))

        image_obj.mask = cropped_mask

        return image_obj


if __name__ == "__main__":
    black_image = np.zeros((100, 100, 3), dtype=np.uint8)

    img = Image(black_image)
