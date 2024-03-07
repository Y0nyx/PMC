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

        return Image(np.array(cropped_image))
    
    def subdivise(self, sub_image_size, overlap_size, transformation_type):

        sub_images = []
        # Apply the specified transformation
        if transformation_type == "untranslated":
            sub_images = self._create_untranslated_sub_images(sub_image_size, overlap_size)
        elif transformation_type == "translated_horizontal":
            sub_images = self._create_translated_sub_images(sub_image_size, overlap_size, axis="horizontal")
        elif transformation_type == "translated_vertical":
            sub_images = self._create_translated_sub_images(sub_image_size, overlap_size, axis="vertical")

        return sub_images

    #TODO: fix that shit
    def _create_untranslated_sub_images(self, sub_image_size, overlap_size):
        # Get the dimensions of the original image
        image = self.value
        channels, width, height = self.get_size()
        # Calculate the number of sub-images in both dimensions
        num_sub_images_x = width // sub_image_size
        num_sub_images_y = height // sub_image_size

        sub_image_list = []

        # Iterate over the sub-images and save each one with overlap
        for i in range(num_sub_images_x):
            for j in range(num_sub_images_y):
                left = i * sub_image_size
                top = j * sub_image_size
                right = left + sub_image_size
                bottom = top + sub_image_size

                #TODO: ajouter fonction overlap si jamais on pense que c'est encore pertinent
                #left, top, right, bottom = add_overlap(left, top, right, bottom, width, height, overlap_size)
                # Crop and save the sub-image with overlap
                sub_image = image[left:right, top:bottom]
                sub_image_list.append(Image(sub_image))

        return sub_image_list


    def _create_translated_sub_images(self, image, output_folder, overlap_size, axis):
        # Get the dimensions of the original image
        width, height = image.size
        # Calculate the number of sub-images in both dimensions
        num_sub_images_x = width // sub_image_size
        num_sub_images_y = height // sub_image_size
        if(axis=="horizontal"): num_sub_images_x-=1
        else: num_sub_images_y-=1
        # Iterate over the sub-images and save each one with overlap
        for i in range(num_sub_images_x):
            for j in range(num_sub_images_y):
                left = i * sub_image_size
                top = j * sub_image_size
                right = left + sub_image_size
                bottom = top + sub_image_size
                # Add overlap pixels from the original image based on the specified axis
                if axis == "horizontal":
                    left += sub_image_size/2
                    right += sub_image_size/2
                elif axis == "vertical":
                    top += sub_image_size/2
                    bottom += sub_image_size/2
                else:
                    raise ValueError("Invalid axis. Use 'horizontal' or 'vertical'.")
                left, top, right, bottom = add_overlap(left, top, right, bottom, width, height, overlap_size)
                # Crop and save the sub-image with overlap
                sub_image = image.crop((left, top, right, bottom))
                output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_translated_{axis}_{i}_{j}_overlap.png")
                sub_image.save(output_path)
