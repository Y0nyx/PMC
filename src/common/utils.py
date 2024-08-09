import os
import warnings
import numpy as np
from pathlib import Path
from common.image.Image import Image
from common.image.ImageCollection import ImageCollection


class DataManager:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.current_iteration = 0

        self.iteration_dirs = sorted(
            item
            for item in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, item))
        )

        if not self.iteration_dirs:
            warnings.warn("This path is empty")

    def get_all_img(self, generate_mask = False) -> ImageCollection:
        if len(self.iteration_dirs) > self.current_iteration:
            current_iteration_dir = self.dataset_path / Path(
                self.iteration_dirs[self.current_iteration]
            )

            image_files = current_iteration_dir.glob("*.jpg")
            images = [Image(img_path) for img_path in image_files]
            self.current_iteration += 1

            if generate_mask is True:
                bounding_boxes = load_bounding_boxes_from_folder(current_iteration_dir)

                # Process each label file and create/save the corresponding mask
                for i, label_file_name, boxes in enumerate(bounding_boxes):
                    mask = create_bounding_box_mask(label_file_name, boxes, image_width, image_height)
                    images[i].mask = mask
                    save_mask_image(mask, current_iteration_dir+'/original_masks', label_file_name)

            return ImageCollection(images)

    def load_bounding_boxes_from_folder(folder_path):
        """
        Load bounding box information from text files in a specified folder.

        Args:
        - folder_path: Path to the folder containing the text files.

        Returns:
        - List of tuples, where each tuple contains the label file name and a list of tuples of (class, x_center, y_center, width, height) of bounding boxes in YOLO format.
        """
        bounding_boxes = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                boxes = []
                with open(os.path.join(folder_path, file_name), 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        class_, x_center, y_center, width, height = map(float, parts)
                        boxes.append((class_, x_center, y_center, width, height))
                bounding_boxes.append((file_name, boxes))
        return bounding_boxes

    def create_bounding_box_mask(label_file_name, bounding_boxes, image_width, image_height):
        """
        Create a mask image of the image where the bounding boxes are.

        Args:
        - label_file_name: Name of the label file.
        - bounding_boxes: List of tuples, where each tuple contains (class, x_center, y_center, width, height) of bounding boxes in YOLO format.
        - image_width: Width of the image.
        - image_height: Height of the image.

        Returns:
        - Numpy array representing the mask of the bounding boxes.
        """
        # Initialize an empty mask with the same dimensions as the image
        mask = np.zeros((image_height, image_width))

        if not bounding_boxes:
            return mask

        for bbox in bounding_boxes:
            class_, x_center, y_center, width, height = bbox

            # Convert YOLO format to Pascal VOC format
            x_min = max(0, (x_center - width / 2) * image_width)
            y_min = max(0, (y_center - height / 2) * image_height)
            x_max = min(image_width, (x_center + width / 2) * image_width)
            y_max = min(image_height, (y_center + height / 2) * image_height)

            # Ensure the bounding box coordinates are within the image dimensions
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_width, x_max)
            y_max = min(image_height, y_max)

            # Set the mask values within the bounding box to 1
            mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 255

        return mask
