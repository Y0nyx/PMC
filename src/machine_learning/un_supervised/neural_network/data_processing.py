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
import re

from random import randint, uniform
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse_perimeter
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from PIL import Image as Img

import os
from pathlib import Path

class DataProcessing():

    def __init__(self, height, width, seg_model_path: str ="/home/jean-sebastien/Documents/s7/PMC/PMC/src/ia/segmentation/v2.pt"):  #"../../../../src/ia/segmentation/v2.pt" #TODO Convert for docker
        self.height = height
        self.width = width
        self._nb_x_sub = 0
        self._nb_y_sub = 0
        self.debug = False
        self._segmentation_model = YOLO(Path(seg_model_path))

    def load_data(self, data_path, subdvisise=False, segment=False, rotate=False, test=False):

        if not test:
            images = []
            sorted_filenames = sorted(os.listdir(data_path), key=self.custom_sort_key)
            #Data loading
            for filename in sorted_filenames:
                if filename.endswith(".jpg"):
                    images.append(cv2.imread(f'{data_path}/{filename}'))
        
        else:
            data_path = '/home/jean-sebastien/Documents/s7/PMC/Data/4k_dataset/original_images/sans_defauts_test_blanc' #TODO Convert for docker
            images = []
            sorted_filenames = sorted(os.listdir(data_path), key=self.custom_sort_key)
            #Data loading
            for filename in sorted_filenames:
                if filename.endswith(".jpg"):
                    images.append(cv2.imread(f'{data_path}/{filename}'))

            data_path_test = "/home/jean-sebastien/Documents/s7/PMC/Data/4k_dataset/original_images/avec_defauts_full_data_blanc" #TODO Convert for docker
            defauts_images = []
            sorted_filenames_defauts = sorted(os.listdir(data_path_test), key=self.custom_sort_key)
            for filename in sorted_filenames_defauts:
                if filename.endswith(".jpg"):
                    defauts_images.append(cv2.imread(f'{data_path_test}/{filename}'))

            print(f'there are {len(images)} without defect')
            print(f'there are {len(defauts_images)} without defect')

        rotated_images = []
        sub_images = []
        seg_images = []

        # for i, image in enumerate(images):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

        #             # Ensure the output directory exist
        
        if segment:
            print('Segmenting the image')
            for i, image in enumerate(images):
                print(f'This is the variable type {type(image)}')
                print(f'This is the shape of the image YOLO {image.shape}')
                seg_images.extend(self.segment(image))

            images = seg_images 

            if test:
                seg_images_defauts = []

                for i, image in enumerate(defauts_images):
                    seg_images_defauts.extend(self.segment(image))

                defauts_images = seg_images_defauts
            
            if self.debug and not test:
                self.save_images(len(images), images, "/home/jean-sebastien/Documents/s7/PMC/PMC/src/machine_learning/un_supervised/output_segmentation_soudure") #"./output_segmentation" TODO Convert for docker
            elif self.debug and test:
                self.save_images(len(defauts_images), defauts_images, "/home/jean-sebastien/Documents/s7/PMC/PMC/src/machine_learning/un_supervised/output_segmentation_soudure_defauts") #"./output_segmentation" TODO Convert for docker

        if subdvisise:
            print('Subdivising the image')
            for i, img in enumerate(images):
                sub_images.extend(self.subdivise(img))
            images = sub_images
            print(f'The number of subdivised images is {len(images)}')

            if test:
                sub_images_default = []
                for i, img in enumerate(defauts_images):
                    sub_images_default.extend(self.subdivise(img))
                defauts_images = sub_images_default
                print(f'The number of subdivised images is {len(defauts_images)}')

            if self.debug:
                self.save_images(len(images), images, "/home/jean-sebastien/Documents/s7/PMC/PMC/src/machine_learning/un_supervised/output_segmentation") #"./output_segmentation" TODO Convert for docker

        if rotate:
            for i, image in enumerate(images):
                rotated_images.extend(self.rotate(image))

            images = rotated_images

        if not test:
            #Convert the image back to rgb
            for i, image in enumerate(images):
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            images = np.array(images)

            print('Data processing has been completed.')
            return self.split_data(images)
        else:
            non_defauts_images = np.array(images)
            defauts_images = np.array(defauts_images)
            
            print('Data processing has been completed.')
            return non_defauts_images, defauts_images


    def split_data(self, images):
        """
        Split the data into three set. Training set(70-80%), Validation set (10-15%) and Test set (10-15%)
        """

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
    
    def add_stain(self, image, max_pixel_value):
        elipse_size = "5-10"
        blur = 0
        pourcentage_scale_factor = 100.

        row, column = image.shape[0], image.shape[1]
        min_range, max_range = float(elipse_size.split('-')[0]), float(elipse_size.split('-')[1])
        a, b = randint(int(min_range/pourcentage_scale_factor*column), int(max_range/pourcentage_scale_factor*column)), randint(int(min_range/pourcentage_scale_factor*row), int(max_range/pourcentage_scale_factor*row))
        rotation = uniform(0, 2*np.pi)

        cx, cy = randint(max(a,b), int(column-max(a,b))), randint(max(a,b), int(row-max(a,b)))
        x, y = ellipse_perimeter(cy, cx, a, b, rotation)
        contour  = np.array([[i,j] for i,j in zip(x,y)])

        mask = np.zeros((row, column))
        mask = cv2.drawContours(mask, [contour], -1, 1, -1)

        if blur != 0:
            mask = gaussian_filter(mask, max(a,b)*blur)

        color = image[cy, cx].astype('float32') / float(max_pixel_value)  # Normaliser la couleur

        rgb_mask = np.dstack([mask]*3)

        not_modified = np.subtract(np.ones(image.shape), rgb_mask)
        stain = color * rgb_mask  # Utilisez la couleur sélectionnée pour la tache
        image_stain = np.add(np.multiply(image, not_modified), np.multiply(stain, rgb_mask))

        return image_stain

    def apply_random_blackout(self, images, blackout_size=(256, 256)):
        """
        Apply random blackout to data for self supervised training. 
        """
        augmented_images = images.copy()

        # Randomly select the position to blackout
        x = np.random.randint(0, images.shape[0] - blackout_size[0] + 1)
        y = np.random.randint(0, images.shape[1] - blackout_size[1] + 1)

        # Black out the selected region for each channel
        channels = 3
        for channel in range(channels):
            augmented_images[x:x+blackout_size[0], y:y+blackout_size[1], channel] = 0.0

        return augmented_images
    
    def get_random_blackout(self, input_train, input_valid):
        """
        Call apply_random_blackout for every data that need to be data augmented. 
        """
        train_augmented = self.apply_random_blackout(input_train)
        valid_augmented = self.apply_random_blackout(input_valid)

        print("=====================================")
        print("Augmented splitted dataset in arrays of shape: ", train_augmented.shape, " | ",valid_augmented.shape)
        print("=====================================")



        return train_augmented, valid_augmented

    def save_images(self, nb_images, images, folder: str = "./output_"):
        output_dir = folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(nb_images):
         cv2.imwrite(f"{output_dir}/output_{i}.png", images[i])
        
    def normalize(self, data, max_pixel_value):
        """
        Data domain will be between 0 and 1. 
        """
        return data.astype('float32') / float(max_pixel_value)
    
    def de_normalize(self, input_test, result_test, max_pixel_value):
        difference_abs = np.abs(input_test - result_test)

        #Normalisation
        min_val = np.min(difference_abs)
        max_val = np.max(difference_abs)

        #Ajustement des valeurs pour couvrir la gamme de 0 à 255
        normalized_diff = (difference_abs - min_val) / (max_val - min_val) * max_pixel_value

        #Conversion en uint8
        difference_reshaped = normalized_diff.astype('uint8')
        input_test = (input_test * max_pixel_value).astype('uint8')
        result_test = (result_test * max_pixel_value).astype('uint8')

        return input_test, result_test, difference_reshaped
    
    def get_data_processing_standard(self, data_path, max_pixel_value, test=False):
        """
        Do the data processing on the original data without any patch. 
        Used for thes 1. 
        """
        print("You are currently doing test 1")
        if not test:
            input_train, input_valid, input_test = self.load_data(data_path, subdvisise=True, segment=True)

            train_input_norm = self.normalize(input_train, max_pixel_value)
            train_input_loss_norm = train_input_norm
            valid_input_norm = self.normalize(input_valid, max_pixel_value)
            valid_input_loss_norm = valid_input_norm
            test_input_norm = self.normalize(input_test, max_pixel_value)

            return train_input_norm, train_input_loss_norm, valid_input_norm, valid_input_loss_norm, test_input_norm
        else:
            non_defauts_images, defauts_images = self.load_data(data_path, subdvisise=False, segment=False, test=True)
            
            # test_input_norm = self.normalize(non_defauts_images, max_pixel_value)
            # del non_defauts_images
            # defaut_images_norm = self.normalize(defauts_images, max_pixel_value)
            # del defauts_images

            return non_defauts_images, defauts_images
        
    def get_data_processing_blackout(self, data_path, max_pixel_value, test=False):
        """
        Do the processing for the data set used by the team to work with. 
        Used for test 2.
        """
        print("You are currently doing test 2")
        if not test:
            input_train, input_valid, input_test = self.load_data(data_path, subdvisise=True, segment=True)

            train_blackout, valid_blackout = self.get_random_blackout(input_train, input_valid)

            train_input_norm = self.normalize(train_blackout, max_pixel_value)
            train_input_loss_norm = self.normalize(input_train, max_pixel_value)
            valid_input_norm = self.normalize(valid_blackout, max_pixel_value) 
            valid_input_loss_norm = self.normalize(input_valid, max_pixel_value)
            test_input_norm = self.normalize(input_test, max_pixel_value)

            if self.debug:
                self.save_images(15, self.de_normalize(train_input_norm, max_pixel_value), "train_input_norm")
                self.save_images(15, self.de_normalize(train_input_loss_norm, max_pixel_value), "train_input_loss_norm")
                self.save_images(15, self.de_normalize(valid_input_norm, max_pixel_value), "valid_input_norm")
                self.save_images(15, self.de_normalize(valid_input_loss_norm, max_pixel_value), "valid_input_loss_norm")
                self.save_images(15, self.de_normalize(test_input_norm, max_pixel_value), "test_input_norm")

            return train_input_norm, train_input_loss_norm, valid_input_norm, valid_input_loss_norm, test_input_norm
        else:
            non_defauts_images, defauts_images = self.load_data(data_path, segment=False, test=True)

            # test_input_norm = self.normalize(input_test, max_pixel_value)
            # defaut_images_norm = self.normalize(defauts_images, max_pixel_value)

            return non_defauts_images, defauts_images
    
    def get_data_processing_stain(self, data_path, max_pixel_value, test=False):
        """
        Do the data processing with Stain noise used to try to beat the bench mark
        Used for test 3.
        """
        print("You are currently doing test 3")
        if not test: 
            input_train, input_valid, input_test = self.load_data(data_path, subdvisise=True, segment=True)

            images_stain_train = []
            images_stain_valid = []
            for img in input_train:
                images_stain_train.append(self.add_stain(img, max_pixel_value)) 
            for img in input_valid:
                images_stain_valid.append(self.add_stain(img, max_pixel_value))
            images_stain_train = np.array(images_stain_train)
            images_stain_valid = np.array(images_stain_valid)

            train_input_norm = self.normalize(images_stain_train, max_pixel_value)    
            train_input_loss_norm = self.normalize(input_train, max_pixel_value)
            valid_input_norm = self.normalize(images_stain_valid, max_pixel_value) 
            valid_input_loss_norm = self.normalize(input_valid, max_pixel_value)
            test_input_norm = self.normalize(input_test, max_pixel_value)

            return train_input_norm, train_input_loss_norm, valid_input_norm, valid_input_loss_norm, test_input_norm
        else:
            non_defauts_images, defauts_images = self.load_data(data_path, segment=False, test=True) 

            # test_input_norm = self.normalize(input_test, max_pixel_value)
            # defaut_images_norm = self.normalize(defauts_images, max_pixel_value)

            return non_defauts_images, defauts_images

    
    def resize(self, image):
        closest_width = int(np.ceil(image.shape[1] / self.width ) * self.width )
        closest_height = int(np.ceil(image.shape[0] / self.height) * self.height)
        print("width", self.width)
        print("width closest", closest_width)
        return cv2.resize(image, (closest_width, closest_height))
    
    def subdivise(self, image):
        print("sub_func")
        sub_images = []
        # Get the dimensions of the original image

        self.resize(image)
        print("resized")

        width, height, channels = image.shape

        # Calculate the number of sub-images in both dimensions
        self._nb_x_sub = width // self.width
        self._nb_y_sub = height // self.height
        print("Sub x: ",self._nb_x_sub)
        print("Sub y: ",self._nb_y_sub)

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
                sub_images.append(image[left:right, top:bottom, :])
        return sub_images
    
    def rotate(self, image):
        rotated_images = [image]
        for _ in range(3):
            image = np.rot90(image)
            rotated_images.append(image)
        return rotated_images
    
    def segment(self, image):
        imgCollection = []
        
        results = self._segmentation_model.predict(source=image, show=False, conf=0.7, save=False)
        # crop images with bounding box
        for result in results:
            for boxes in result.boxes:
                imgCollection.append(self.crop(boxes, image))

        return imgCollection
    
    def crop(self, boxes, image):
        image = Img.fromarray(image, 'RGB')
        cropped_image = image.crop(boxes.xyxy.tolist()[0])
        cropped_image = np.array(cropped_image)
        return cropped_image
    
    def custom_sort_key(self, filename):
        match = re.match(r'(\d+)([a-z]+)_.*', filename)
        if match:
            # Return a tuple with the numeric part as an integer and the prefix
            return (int(match.group(1)), match.group(2))
        else:
            # If no match, return a tuple that puts the filename last
            return (float('inf'), filename)  
