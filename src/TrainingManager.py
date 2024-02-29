import os
from datetime import datetime
from shutil import move
from common.Constants import *


class TrainingManager:
    def __init__(
        self, is_time_threshold: bool = True, is_number_threshold: bool = True
    ) -> None:
        self.is_time_threshold = is_time_threshold
        self.is_number_threshold = is_number_threshold
        self.flags = []
        self.is_ready_training = False

    def check_img_number(self):
        if self.is_number_threshold:
            result = len(os.listdir(IMG_SAVE_FILE)) > NUMBER_IMG_THRESHOLD
            self.flags.append(result)
            if result:
                self.separate_dataset()

    def check_time(self):
        if self.is_time_threshold:
            self.flags.append(datetime.now().hour > TIME_THRESHOLD)

    def check_flags(self):
        if not self.is_ready_training:
            self.flags = []
            self.check_time()
            self.check_img_number()
            self.is_ready_training = all(self.flags)
        return self.is_ready_training

    def separate_dataset(self):
        if self.check_flags():
            # Get a list of image files in IMG_SAVE_FILE
            image_files = os.listdir(IMG_SAVE_FILE)
            num_images = len(image_files)
            num_train = int(num_images * TRAIN_SPLIT)
            num_valid = num_images - num_train

            # Create train and valid directories if they don't exist
            os.makedirs(TRAIN_FILE, exist_ok=True)
            os.makedirs(VALID_FILE, exist_ok=True)

            # Move images to train directory
            for img_file in image_files[:num_train]:
                move(
                    os.path.join(IMG_SAVE_FILE, img_file),
                    os.path.join(TRAIN_FILE, img_file),
                )

            # Move remaining images to valid directory
            for img_file in image_files[num_train:]:
                move(
                    os.path.join(IMG_SAVE_FILE, img_file),
                    os.path.join(VALID_FILE, img_file),
                )
