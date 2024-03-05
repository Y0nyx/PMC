import cv2
import numpy as np

class data_pipeline:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self._subdivisions = []
        self._nb_x_sub = 0
        self._nb_y_sub = 0
    
    def set_image(self, image):
        self._image = image

    def resize(self):
        closest_width = int(np.ceil(self._image.shape[1] / self.width ) * self.width )
        closest_height = int(np.ceil(self._image.shape[0] / self.height) * self.height)
        self._image = cv2.resize(self._image, (closest_width, closest_height))
    
    def subdivise(self):
        print("sub_func")
        sub_images = []
        # Get the dimensions of the original image

        self.resize()
        print("resized")

        width, height, channels = self._image.shape

        # Calculate the number of sub-images in both dimensions
        self._nb_x_sub = width // self._model.input_shape[1]
        self._nb_y_sub = height // self._model.input_shape[2]

        # Iterate over the sub-images and save each one with overlap
        for i in range(self._nb_x_sub):
            for j in range(self._nb_y_sub):
                left = i * self.width
                top = j * self.height
                right = left + self.width
                bottom = top + self.height

                # TODO: Add overlap code
                # left, top, right, bottom = add_overlap(left, top, right, bottom, width, height, overlap_size)

                # Crop the sub-image using NumPy array slicing
                sub_images.append(self._image[left:right, top:bottom, :])

        return sub_images