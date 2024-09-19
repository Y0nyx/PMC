import os
import yaml
from datetime import datetime
from shutil import move
from common.Constants import *
from pipeline.models.Model import YoloModel


class TrainingManager:
    """Class to manage training process."""

    def __init__(
        self, is_time_threshold: bool = True, is_number_threshold: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Initialize TrainingManager.

        Parameters:
        - is_time_threshold (bool): Whether to check time threshold.
        - is_number_threshold (bool): Whether to check image number threshold.
        - verbose (bool): Whether to print verbose output.
        """
        self.verbose = verbose
        self.is_time_threshold = is_time_threshold
        self.is_number_threshold = is_number_threshold
        self.flags = []
        self.is_ready_training = False

    def check_img_number(self):
        """
        Check if the number of images exceeds the threshold.
        """
        if self.is_number_threshold:
            number_files = len(os.listdir(CAPTURED_IMG_SAVE_PATH))
            self.print(f'Check Images Number : {number_files} / {NUMBER_IMG_THRESHOLD}')
            self.flags.append(number_files > NUMBER_IMG_THRESHOLD)

    def check_time(self):
        """
        Check if the current time exceeds the threshold.
        """
        if self.is_time_threshold:
            self.print(f'Check Time : {datetime.now().hour} / {TIME_THRESHOLD}')
            self.flags.append(datetime.now().hour > TIME_THRESHOLD)

    def check_flags(self):
        """
        Check all the flags to determine if the training can proceed.
        """
        if not self.is_ready_training:
            self.print('===== Check Flags =====')
            self.flags = []
            self.check_time()
            self.check_img_number()
            self.print(f"Flags : {self.flags}")
            self.is_ready_training = all(self.flags)
        return self.is_ready_training

    def separate_dataset(self):
        """
        Split the dataset into training and validation sets.
        """
        image_files = os.listdir(CAPTURED_IMG_SAVE_PATH)
        num_images = len(image_files)
        num_train = int(num_images * TRAIN_SPLIT)

        os.makedirs(TRAIN_FILE, exist_ok=True)
        os.makedirs(VALID_FILE, exist_ok=True)

        for img_file in image_files[:num_train]:
            move(
                os.path.join(CAPTURED_IMG_SAVE_PATH, img_file),
                os.path.join(TRAIN_FILE, img_file),
            )

        for img_file in image_files[num_train:]:
            move(
                os.path.join(CAPTURED_IMG_SAVE_PATH, img_file),
                os.path.join(VALID_FILE, img_file),
            )

    def train_supervised(self) -> YoloModel:
        """
        Train the model using supervised learning.
        
        Returns:
        - bool: True if training is successful, False otherwise.
        """
        self.generate_yaml()

        model = YoloModel()
        try:
            model.train(data=YAML_FILE, epochs=EPOCHS, batch=BATCH)
        except Exception as e:
            print(e)
            return None
        
        return model
    
    def generate_yaml(self) -> bool:
        """
        Generate YAML file for training configuration.
        
        Returns:
        - bool: True if YAML generation is successful, False otherwise.
        """
        data = {
            'train': str(TRAIN_FILE.absolute()),
            'val': str(VALID_FILE.absolute()),
            'nc': NC,
            'names': CLASSES
        }

        try:
            with open(YAML_FILE, 'w') as file:
                yaml.dump(data, file)
        except Exception as e:
            print(e)
            return False
        return True
    
    def print(self, string):
        """
        Print string if verbose mode is enabled.
        
        Parameters:
        - string (str): String to print.
        """
        if self.verbose:
            print(string)

if __name__ == '__main__':
    tm = TrainingManager()

    tm.train_supervised()
