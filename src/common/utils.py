import os
import warnings
import numpy as np
from pathlib import Path
from common.image.Image import Image
from common.image.ImageCollection import ImageCollection


class DataManager:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.current_iteration = 1

        self.iteration_dirs = sorted(
            item
            for item in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, item))
        )

        if not self.iteration_dirs:
            warnings.warn("This path is empty")

    def get_all_img(self) -> ImageCollection:
        current_iteration_dir = self.dataset_path / Path(
            self.iteration_dirs[self.current_iteration - 1]
        )

        image_files = current_iteration_dir.glob("*.jpg")
        images = [Image(img_path) for img_path in image_files]
        self.current_iteration += 1

        return ImageCollection(images)
