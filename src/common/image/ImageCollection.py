from dataclasses import dataclass

import os


@dataclass
class ImageCollection:
    def __init__(self, img_list=[]) -> None:
        self._img_list = img_list
        self._save_counter = 0

    def __iter__(self):
        return iter(self._img_list)

    def get_size(self):
        return [img.get_size() for img in self._img_list]

    def resize(self, width, height) -> bool:
        for img in self._img_list:
            if not img.resize(width, height):
                return False
        return True

    def save(self, file_path):
        if self._save_counter == 0:
            self._save_counter = len(os.listdir(file_path))
        for img in self._img_list:
            img.save(file_path)

    def add(self, img):
        self._img_list.append(img)
