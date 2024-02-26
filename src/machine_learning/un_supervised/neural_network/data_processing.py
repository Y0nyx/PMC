#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux and Mathias Gagnon
Contributor(s): 
Date: 02/12/2024
Description: Process the data and split the data into set before Neural Network training.  
"""

import cv2
import math
import numpy as np
import os

from keras.preprocessing import image
from random import randint, uniform
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse_perimeter
from skimage.util import random_noise
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
        len_separation = int( math.floor(len(input_train_valid) * 0.78) ) # 70% Train / 20% Valid / 10% Test.
        input_train = input_train_valid[:len_separation]
        input_valid = input_train_valid[len_separation:]
        print("=====================================")
        print(f'The sata is splitted into three set. TRAIN, VALIDATION and TEST')
        print("Splitted dataset in arrays of shape: ", input_train.shape, " | ", input_valid.shape, " | ", input_test.shape)
        print("=====================================")

        del images

        return input_train, input_valid, input_test
    
    def add_stain(self, image):
        elipse_size = "5-10"
        blur = 0

        row, column = image.shape[0], image.shape[1]
        min_range, max_range = float(elipse_size.split('-')[0]), float(elipse_size.split('-')[1])
        a, b = randint(int(min_range/100.*column), int(max_range/100.*column)), randint(int(min_range/100.*row), int(max_range/100.*row))
        rotation = uniform(0, 2*np.pi)

        cx, cy = randint(max(a,b), int(column-max(a,b))), randint(max(a,b), int(row-max(a,b)))
        x, y = ellipse_perimeter(cy, cx, a, b, rotation)
        contour  = np.array([[i,j] for i,j in zip(x,y)])

        mask = np.zeros((row, column))
        mask = cv2.drawContours(mask, [contour], -1, 1, -1)

        if blur != 0:
            mask = gaussian_filter(mask, max(a,b)*blur)

        # Choisissez une couleur aléatoire de l'image pour la tache
        # Par exemple, prenez la couleur du centre de l'ellipse
        color = image[cy, cx].astype('float32') / 255.  # Normaliser la couleur

        rgb_mask = np.dstack([mask]*3)

        not_modified = np.subtract(np.ones(image.shape), rgb_mask)
        stain = color * rgb_mask  # Utilisez la couleur sélectionnée pour la tache
        image_stain = np.add(np.multiply(image, not_modified), np.multiply(stain, rgb_mask))

        return image_stain

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

    def normalize(self, data):
        """
        Data domain will be between 0 and 1. 
        """
        return data.astype('float32') / 255.
    
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
        
    def get_data_processing_blackout(self, data_path):
        """
        Do the processing for the data set used by the team to work with. 
        Used to do the bench mark.
        """
        input_train, input_valid, input_test = self.split_data(data_path)

        train_augmented, valid_augmented, test_augmented = self.get_random_blackout(input_train, input_valid, input_test)

        input_train_norm, input_valid_norm, input_test_norm = self.normalize(input_train, input_valid, input_test)
        input_train_aug_norm, input_valid_aug_norm, input_test_aug_norm = self.normalize(train_augmented, valid_augmented, test_augmented)

        return input_train_norm, input_valid_norm, input_test_norm, input_train_aug_norm, input_valid_aug_norm, input_test_aug_norm
    
    def get_data_processing_stain(self, data_path):
        """
        Do the data processing with Stain noise used to try to beat the bench mark
        """
        input_train, input_valid, input_test = self.split_data(data_path)

        images_stain = []
        for img in input_train:
            images_stain.append(self.add_stain(img)) 
        images_stain = np.array(images_stain)

        train_input_norm = self.normalize(input_train)
        train_input_loss_norm = self.normalize(images_stain)
        valid_input_norm = self.normalize(input_valid)
        test_input_norm = self.normalize(input_test)

        return train_input_norm, train_input_loss_norm, valid_input_norm, test_input_norm
    