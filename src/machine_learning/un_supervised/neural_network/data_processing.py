#!/usr/bin/env python
"""
Author: Mathias Gagnon and Jean-Sebastien Giroux
Contributor(s): 
Date: 02/12/2024
Description: Process the data and split the data into set before Neural Network training.  
"""

import math
import numpy as np
import os

from keras.preprocessing import image
from sklearn.model_selection import train_test_split

class DataProcessing():
    def split_data(self, data_path):
        """
        Split the data into three set. Training set(70-80%), Validation set (10-15%) and Test set (10-15%)
        """
        images = []

        #Data loading
        for filename in os.listdir(data_path):
            if filename.endswith(".png"):
                img = image.load_img(f'{data_path}/{filename}', target_size=(256, 256))
                images.append(image.img_to_array(img))
        images = np.array(images)

        print("=====================================")
        print("Loaded image np.array of shape: ", images.shape)
        print("=====================================")

        # Split the dataset into training and testing sets (90/10 split)
        input_train_valid, input_test = train_test_split(images, train_size=0.9, test_size=0.1, random_state=59)
        #Create a validation set
        len_separation = int( math.floor(len(input_train_valid) * 0.91) )
        input_train = input_train_valid[:len_separation]
        input_valid = input_train_valid[len_separation:]
        print("=====================================")
        print(f'The sata is splitted into three set. TRAIN, VALIDATION and TEST')
        print("Splitted dataset in arrays of shape: ", input_train.shape, " | ", input_valid.shape, " | ", input_test.shape)
        print("=====================================")

        del images

        return input_train, input_valid, input_test

    def apply_random_blackout(self, images, blackout_size=(32, 32)):
        """
        Apply random blackout to data in order to simulate data defect. 
        """
        augmented_images = images.copy()

        for i in range(images.shape[0]):
            # Randomly select the position to blackout
            x = np.random.randint(0, images.shape[1] - blackout_size[0] + 1)
            y = np.random.randint(0, images.shape[2] - blackout_size[1] + 1)

            # Black out the selected region for each channel
            channels = 3
            for channel in range(channels):
                augmented_images[i, x:x+blackout_size[0], y:y+blackout_size[1], channel] = 0.0

        return augmented_images
    
    def get_random_blackout(self, input_train, input_valid, input_test):
        """
        Call apply_random_blackout for every data that need to be data augmented. 
        """
        train_augmented = self.apply_random_blackout(input_train)
        valid_augmented = self.apply_random_blackout(input_valid)
        test_augmented = self.apply_random_blackout(input_test)
        print("=====================================")
        print("Augmented splitted dataset in arrays of shape: ", train_augmented.shape, " | ",valid_augmented.shape, " | ", test_augmented.shape)
        print("=====================================")

        return train_augmented, valid_augmented, test_augmented

    def normalize(self, input_train, input_valid, input_test):
        """
        Data domain will be between 0 and 1. 
        """
        input_train = input_train.astype('float32') / 255.
        input_valid = input_valid.astype('float32') / 255.
        input_test = input_test.astype('float32') / 255.

        return input_train, input_valid, input_test
    
    def de_normalize(self, input_test, result_test):
        difference_abs = np.abs(input_test - result_test)

        #Normalisation
        min_val = np.min(difference_abs)
        max_val = np.max(difference_abs)

        #Ajustement des valeurs pour couvrir la gamme de 0 à 255
        normalized_diff = (difference_abs - min_val) / (max_val - min_val) * 255

        #Conversion en uint8
        difference_reshaped = normalized_diff.astype('uint8')
        input_test = (input_test * 255).astype('uint8')
        result_test = (result_test * 255).astype('uint8')

        return input_test, result_test, difference_reshaped
        
    def get_data_processing(self, data_path):
        """
        Do the processing for the data set used by the team to work with. 
        """
        input_train, input_valid, input_test = self.split_data(data_path)

        train_augmented, valid_augmented, test_augmented = self.get_random_blackout(input_train, input_valid, input_test)

        input_train_norm, input_valid_norm, input_test_norm = self.normalize(input_train, input_valid, input_test)
        input_train_aug_norm, input_valid_aug_norm, input_test_aug_norm = self.normalize(train_augmented, valid_augmented, test_augmented)

        return input_train_norm, input_valid_norm, input_test_norm, input_train_aug_norm, input_valid_aug_norm, input_test_aug_norm