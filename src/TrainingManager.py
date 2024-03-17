import os
import socket
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
            number_files = len(os.listdir(IMG_SAVE_FILE))
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
        image_files = os.listdir(IMG_SAVE_FILE)
        num_images = len(image_files)
        num_train = int(num_images * TRAIN_SPLIT)

        os.makedirs(TRAIN_FILE, exist_ok=True)
        os.makedirs(VALID_FILE, exist_ok=True)

        for img_file in image_files[:num_train]:
            move(
                os.path.join(IMG_SAVE_FILE, img_file),
                os.path.join(TRAIN_FILE, img_file),
            )

        for img_file in image_files[num_train:]:
            move(
                os.path.join(IMG_SAVE_FILE, img_file),
                os.path.join(VALID_FILE, img_file),
            )

    def train_supervised(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            self.print(f"Envoi du signal 'OK' à {SUPERVISED_SERVICE}:{SUPERVISED_PORT}...")
            s.connect((SUPERVISED_SERVICE, SUPERVISED_PORT))
            s.sendall(b'OK')

            # Attendre la réponse
            data = s.recv(1024)
            if data.decode() == 'OK':
                print("Signal de retour 'OK' reçu, entraînement supervisé fini")
                return True
            else:
                return False
    
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
