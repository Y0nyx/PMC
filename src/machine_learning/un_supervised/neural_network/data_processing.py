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
        elipse_size = "10-30"
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

    def apply_random_blackout(self, images, blackout_size=(128, 128)):
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
    
    def normalize_classification(self, data, max_pixel_value):
        """
        Data domain will be between 0 and 1. 
        Used when normalizing the classification data, there are 4 channels in this case, 
        with one channel representing the defect position which will not be normalized.
        """
        # Select only the three first channels, the data is in the format: [sample, width, length, channel]
        data[:, :, :, :3] = data[:, :, :, :3].astype('float32') / float(max_pixel_value)
        return data
    
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
        print("You are currently doing the data processing with stain")
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
        
    def segment_PMC860(self, data_path, type, dataset, output_path):
        """
        segment the dataset into the desired dimension from the original image
        """
        # Loading the data to segment
        data_path = f'{data_path}/{dataset}/segmentation/{type}'
        input_train = []
        filenames = []

        for i, filename in enumerate(os.listdir(data_path)):
            print(f'Loading image: {i+1}')
            if i <= 200:
                if filename.endswith(".jpg"):
                    input_train.append(cv2.imread(f'{data_path}/{filename}'))
                    filenames.append(filename)

        # Segmenting the training and validation data.
        for i, (image_training, filename) in enumerate(zip(input_train, filenames)):
            print(f'Segmenting image: {i+1}')
            seg_images = self.subdivise(image_training)
            for i, img in enumerate(seg_images):
                if not np.all(img == 0):
                    cv2.imwrite(f'{output_path}/{filename}_{i}', img)
        

    def get_data_processing_stain_PMC860(self, data_path, max_pixel_value, testing, test=False):
        """
        Data processing to obtain stain on the data and normalize the data between 0 and 1. 
        Called PMC860, because this dataprocessing is done using the new dataset nammed: "Datasets_segmentation_grayscale"
        """
        print("You are currently doing the data processing with stain for dataset: Datasets_segmentation_grayscale")
        if not test: 
            """
            Setps done during the training phase. 
            """
            # --------------------------------------------------------------------------------------------------------------------
            #                            1. L0ADING THE TEST DATA WITH AND WITOUT DEFECTS
            # --------------------------------------------------------------------------------------------------------------------
            # Loading the training data 
            print(f'Loading the training images')
            input_train_path = f'{data_path}/train/segmentation/as_no_default_test'
            input_train = []

            for i, filename in enumerate(os.listdir(input_train_path)):
                print(f'Loading image: {i}')
                if filename.endswith(".jpg") and i <= 20:
                    img_train = cv2.imread(f'{input_train_path}/{filename}')
                    print(f'The shape for this input is {img_train.shape}')
                    if img_train.shape[0] == 2160 and img_train.shape[1] == 3840:
                        input_train.append(img_train)

            # Loading the validation data
            print(f'Loading the validation images')
            input_valid_path = f'{data_path}/valid/segmentation/as_no_default_test'
            input_valid = []

            for i, filename in enumerate(os.listdir(input_valid_path)):
                print(f'Loading image: {i}')
                if filename.endswith(".jpg") and i <= 20:
                    img_valid = cv2.imread(f'{input_valid_path}/{filename}')
                    print(f'The shape for this input is {img_valid.shape}')
                    if img_valid.shape[0] == 2160 and img_valid.shape[1] == 3840:
                        input_valid.append(img_valid)

            print(f'There are {len(input_train)} input_train and {len(input_valid)} input_valid.')

            # --------------------------------------------------------------------------------------------------------------------
            #                            2. SEGMENT THE IMAGES TO KEEP ONLY THE WELDING
            # --------------------------------------------------------------------------------------------------------------------
            seg_images_training = []
            seg_images_validation = []
            images_bounding_boxes_train = []
            images_bounding_boxes_valid = []
            input_train = np.array(input_train)
            input_valid = np.array(input_valid)

            for i, image in enumerate(input_train):
                print(f'Analysing image {i+1}')
                # Segment the original image
                img1 = image.copy()
                bounding_boxes = self.segment_welds_opencv(img1, train=True)
                # Add the bounding box to the original image
                image_bounding_boxes = self.draw_bounding_boxes(img1, bounding_boxes, train=True)
                print(f'The shape for the bounding box is: {image_bounding_boxes.shape}')
                images_bounding_boxes_train.append(image_bounding_boxes)

                # Crop the original image
                cropped_images = self.crop_weld_sections(image, bounding_boxes)
                seg_images_training.extend(cropped_images)

            print(f'\nThere are: {len(seg_images_training)} images.')
            
            for i, image in enumerate(input_valid):
                print(f'Analyzing image {i+1}')
                # Segment the original image
                img2 = image.copy()
                bounding_boxes = self.segment_welds_opencv(img2, train=True)
                # Add the bounding box to the original image
                image_bounding_boxes = self.draw_bounding_boxes(img2, bounding_boxes, train=True)
                print(f'The shape for the bounding box is: {image_bounding_boxes.shape}')
                images_bounding_boxes_valid.append(image_bounding_boxes)

                # Crop the original image
                cropped_images = self.crop_weld_sections(image, bounding_boxes)
                seg_images_validation.extend(cropped_images)

            print(f'There are: {len(seg_images_validation)} images.')

            # **************************************************************************************************
            # ********************************************** TEST **********************************************
            # **************************************************************************************************
            if testing:
                path = f'{data_path}/train/segmentation/a_bounding_boxes'
                if not os.path.exists(path):
                    os.makedirs(path)
                # Save the images to see what the bounding boxes looks like
                for i, img in enumerate(images_bounding_boxes_train):
                    img_path = os.path.join(path, f'weld_{i+1}.png')
                    cv2.imwrite(img_path, img)

                path = f'{data_path}/valid/segmentation/a_bounding_boxes'
                if not os.path.exists(path):
                    os.makedirs(path)
                for i, img in enumerate(images_bounding_boxes_valid):
                    img_path = os.path.join(path, f'weld_{i+1}.png')
                    cv2.imwrite(img_path, img)

                # Save the images to see if the cropping works
                path = f'{data_path}/train/segmentation/b_cropping'
                if not os.path.exists(path):
                    os.makedirs(path)
                for i, img in enumerate(seg_images_training):
                    img = np.array(img)
                    img_path = os.path.join(path, f'weld_{i+1}.png')
                    cv2.imwrite(img_path, img)
                path = f'{data_path}/valid/segmentation/b_cropping'
                if not os.path.exists(path):
                    os.makedirs(path)
                for i, img in enumerate(seg_images_validation):
                    img = np.array(img)
                    img_path = os.path.join(path, f'weld_{i+1}.png')
                    cv2.imwrite(img_path, img)

            # --------------------------------------------------------------------------------------------------------------------
            #                                        3. SUBDIVISE THE CROPPED IMAGES
            # --------------------------------------------------------------------------------------------------------------------
            sub_images_training = []
            sub_images_validating = []

            print(f'Subdivising the no defect images')
            for images_training in seg_images_training:
                if not np.all(images_training == 0):
                    sub_images_training.extend(self.subdivise(images_training))
            input_train = sub_images_training
            print(f'\n\nThere are: {len(input_train)} subdivised images without any defect that are not all 0s (black pixels)\n\n')

            print(f'Subdivising the defect images')
            for images_validating in seg_images_validation:
                if not np.all(images_validating == 0):
                    sub_images_validating.extend(self.subdivise(images_validating))
            input_valid = sub_images_validating
            print(f'\n\nThere are: {len(input_valid)} subdivised images with defect that are not all 0s (black pixels)\n\n')

            # **************************************************************************************************
            # ********************************************** TEST **********************************************
            # **************************************************************************************************
            if testing:
                # Save the images to see if the cropping works
                path = f'{data_path}/train/segmentation/c_subdivise'
                if not os.path.exists(path):
                    os.makedirs(path)
                for i, img in enumerate(input_train):
                    img = np.array(img)
                    print(f'The image shape is: {img.shape}')
                    img_path = os.path.join(path, f'weld_{i+1}.png')
                    cv2.imwrite(img_path, img)
                path = f'{data_path}/valid/segmentation/c_subdivise'
                if not os.path.exists(path):
                    os.makedirs(path)
                for i, img in enumerate(input_valid):
                    img = np.array(img)
                    img_path = os.path.join(path, f'weld_{i+1}.png')
                    cv2.imwrite(img_path, img)

            input_train = np.array(input_train)
            input_valid = np.array(input_valid)
            print(f'\n\nThe shape for input_train is: {input_train.shape} amd the shape for input_valid is: {input_valid.shape}\n\n')

            # --------------------------------------------------------------------------------------------------------------------
            #                               4. DELETE THE IMAGES THAT HAVE ALL PIXELS EQUALS TO 0
            # --------------------------------------------------------------------------------------------------------------------
            print(f'Deleting black images')
            filtered_input_train = []
            cptr = 0
            threshold = 0.65 

            for i, train in enumerate(input_train):
                # Flattening the first 3 channels (ignoring the defect channel)
                pixel_count = np.prod(train.shape)
                # Count the number of zero pixels in the first 3 channels
                zero_count = np.sum(train == 0)
                # Calculate the percentage of zero pixels
                zero_percentage = zero_count / pixel_count
                # If less than threshold, keep the image
                if zero_percentage <= threshold:
                    filtered_input_train.append(train)
                else:
                    cptr += 1

            print(f'We removed: {cptr} no defects images')
            print(f'There are: {len(filtered_input_train)} images without defects after deleting black images')
                    
            filtered_input_valid = []
            cptr = 0 

            for i, test in enumerate(input_valid):
                # Flattening the first 3 channels (ignoring the defect channel)
                pixel_count = np.prod(test.shape)
                # Count the number of zero pixels in the first 3 channels
                zero_count = np.sum(test == 0)
                # Calculate the percentage of zero pixels
                zero_percentage = zero_count / pixel_count
                # If less than 80% of pixels are zeros, keep the image
                if zero_percentage <= threshold:
                    filtered_input_valid.append(test)
                else:
                    cptr += 1

            print(f'We removed: {cptr} defects images')
            print(f'There are: {len(filtered_input_valid)} images with defects after deleting black images')

            filtered_input_train = np.array(filtered_input_train)
            filtered_input_valid = np.array(filtered_input_valid)
            print(f'The shape for filtered_no_defects is: {filtered_input_train.shape} and for filtered_defects is: {filtered_input_valid.shape}')

            # **************************************************************************************************
            # ********************************************** TEST **********************************************
            # **************************************************************************************************
            if testing:
                # Save the images to see if the cropping works
                path = f'{data_path}/train/segmentation/d_delete_0s'
                if not os.path.exists(path):
                    os.makedirs(path)
                for i, img in enumerate(filtered_input_train):
                    img_path = os.path.join(path, f'weld_{i+1}.png')
                    cv2.imwrite(img_path, img)
                path = f'{data_path}/valid/segmentation/d_delete_0s'
                if not os.path.exists(path):
                    os.makedirs(path)
                for i, img in enumerate(filtered_input_valid):
                    img = np.array(img)
                    img_path = os.path.join(path, f'weld_{i+1}.png')
                    cv2.imwrite(img_path, img)

        #     # --------------------------------------------------------------------------------------------------------------------
        #     #                          5. ADDING PATCHES TO THE IMAGES
        #     # --------------------------------------------------------------------------------------------------------------------
        #     images_stain_train = []
        #     images_stain_valid = []
        #     print(f'Adding patches for the training data')
        #     for img in input_train:
        #         images_stain_train.append(self.add_stain(img, max_pixel_value)) 
        #     print(f'Adding patches for the validation data\n')
        #     for img in input_valid:
        #         images_stain_valid.append(self.add_stain(img, max_pixel_value))
        #     images_stain_train = np.array(images_stain_train)
        #     images_stain_valid = np.array(images_stain_valid)
        #     input_train = np.array(input_train)
        #     input_valid = np.array(input_valid)

        #     # **************************************************************************************************
        #     # ********************************************** TEST **********************************************
        #     # **************************************************************************************************



        #     # --------------------------------------------------------------------------------------------------------------------
        #     #                          5. NORMALIZE THE INPUT (VALUES BETWEEN 0-1)
        #     # --------------------------------------------------------------------------------------------------------------------
        #     print(f'Normalizing the data')
        #     train_input_norm = self.normalize(filtered_train_input, max_pixel_value)    
        #     train_input_loss_norm = self.normalize(filtered_train_target, max_pixel_value)
        #     valid_input_norm = self.normalize(filtered_valid_input, max_pixel_value) 
        #     valid_input_loss_norm = self.normalize(filtered_valid_target, max_pixel_value)

        #     return np.array(train_input_norm), np.array(train_input_loss_norm), np.array(valid_input_norm), np.array(valid_input_loss_norm)
        # else:
        #     non_defauts_images, defauts_images = self.load_data(data_path, segment=False, test=True) 

        #     # test_input_norm = self.normalize(input_test, max_pixel_value)
        #     # defaut_images_norm = self.normalize(defauts_images, max_pixel_value)

            return non_defauts_images, defauts_images
        
    def get_data_processing_stain_PMC860_test(self, data_path, max_pixel_value):
        """
        Data processing to obtain stain on the data and normalize the data between 0 and 1. 
        Called PMC860, because this dataprocessing is done using the new dataset nammed: "Datasets_segmentation_grayscale"
        Used when testing the model with test data. 
        Using when doing the regression (Image regeneration) in test.py. 
        """
        print("You are currently doing the data processing with stain for dataset: Datasets_segmentation_grayscale")
        # Loading the training data
        print(f'Loading the testing without defect data')
        input_test_no_defects = f'{data_path}/test/segmentation/as_no_default'
        test_no_defects = []

        for i, filename in enumerate(os.listdir(input_test_no_defects)):
            print(f'Loading image: {i}')
            if filename.endswith(".jpg"):
                test_no_defects.append(cv2.imread(f'{input_test_no_defects}/{filename}'))

        # Loading the validation data
        print(f'Loading the testing with defect data')
        input_test_defects = f'{data_path}/test/segmentation/as_default'
        test_defects = []

        for i, filename in enumerate(os.listdir(input_test_defects)):
            print(f'Loading image: {i}')
            if filename.endswith(".jpg") and i <= 300:
                test_defects.append(cv2.imread(f'{input_test_defects}/{filename}'))

        # Subdivise the training and validation data. 
        sub_images_no_defects = []
        sub_images_defects = []

        print(f'Subdivising the training data')
        for images_training in test_no_defects:
            if not np.all(images_training == 0):
                sub_images_no_defects.extend(self.subdivise(images_training))
        test_no_defects = sub_images_no_defects
        print(f'There are: {len(test_no_defects)} images in subdivise test no defects')

        print(f'Subdivising the validation data\n')
        for images_validating in test_defects:
            if not np.all(images_validating == 0):
                sub_images_defects.extend(self.subdivise(images_validating))
        test_defects = sub_images_defects
        print(f'There are: {len(test_defects)} images in subdivise test defects')

        test_no_defects = np.array(test_no_defects)
        test_defects = np.array(test_defects)

        # Normalizing the input data in range 0 to 1. 
        print(f'Normalizing the data')
        train_no_defects_loss_norm = self.normalize(test_no_defects, max_pixel_value)
        valid_defects_loss_norm = self.normalize(test_defects, max_pixel_value)
        print(f'The shape after the normalization is: {train_no_defects_loss_norm.shape} without defects')
        print(f'The shape after the normalization is: {valid_defects_loss_norm.shape} with defects')

        # Delete images that have all pixels equals to 0. 
        print(f'Deleting black images')
        filtered_no_defects = []
        cptr = 0

        for i, testing_no_defects in enumerate(train_no_defects_loss_norm):
            if not np.all(testing_no_defects == 0):
                filtered_no_defects.append(testing_no_defects)
            else:
                cptr += 1
        print(f'We removed: {cptr} no defects images')
        print(f'There are: {len(filtered_no_defects)} images without defects after deleting black images')
                
        filtered_defects = []
        cptr = 0 

        for i, testing_defects in enumerate(valid_defects_loss_norm):
            if not np.all(testing_defects == 0):
                filtered_defects.append(testing_defects)
            else:
                cptr += 1
        print(f'We removed: {cptr} defects images')
        print(f'There are: {len(filtered_defects)} images with defects after deleting black images')
            
        return np.array(filtered_no_defects), np.array(filtered_defects)


    def get_data_processing_stain_PMC860_test_classification(self, data_path, max_pixel_value, test):
        """
        Data processing to obtain stain on the data and normalize the data between 0 and 1. 
        Called PMC860, because this dataprocessing is done using the new dataset nammed: "Datasets_segmentation_grayscale"
        Used when testing the model with test data. 
        Used when doing the classification in test.py. 
        """
        print("You are currently doing the data processing with stain for dataset: Datasets_segmentation_grayscale")

        # --------------------------------------------------------------------------------------------------------------------
        #                            1. L0ADING THE TEST DATA WITH AND WITOUT DEFECTS
        # --------------------------------------------------------------------------------------------------------------------
        # Loading the test data without defects
        print(f'Loading the testing without defect data')
        no_defect_type = 'no_defect_type_I'
        input_test_no_defects = f'{data_path}/{no_defect_type}/a_original_data'
        test_no_defects = []
        image_nb = 100 # This is a limit to not bust the memory when running on local computer i <= 300

        for i, filename in enumerate(sorted(os.listdir(input_test_no_defects), key=self.custom_sort_key2)):
            print(f'Loading image: {i}, nammed: {filename}')
            if filename.endswith(".jpg") and i <= image_nb:
                no_defect_img = cv2.imread(f'{input_test_no_defects}/{filename}')
                print(f'The shape of the image is: {no_defect_img.shape}')
                # Add a 0 channel (To know where the defect are) 
                # i.e. since we are doing no defect images, we just add a channel with 0 to every pixels. 
                zero_array_shape = list(no_defect_img.shape)
                zero_array_shape[2] = 1
                zero_channel = np.zeros(zero_array_shape)
                print(f'The shape of the zero_channel is: {zero_channel.shape}')
                no_defect_annotated = np.concatenate((no_defect_img, zero_channel), axis=-1)
                # # Test if the last channel are all 0, maybe unittest later?. 
                # print(f'The shape of the no_defect_annotated is: {no_defect_annotated.shape}')
                # if not np.all(no_defect_annotated[:, :, -1] == 0):
                #     print(f'The 0s channels are not all 0.')
                # else:
                #     print(f'The 0s channel are all 0.')
                print(f'\n\n The data shape without defect is: {no_defect_annotated.shape}\n\n')
                test_no_defects.append(no_defect_annotated)
        
        # Loading the test data with defects
        print(f'Loading the testing with defect data')
        defect_type = 'defect_type_I'
        input_test_defects = f'{data_path}/{defect_type}/a_original_data'
        test_defects = []
        lacalisation_defect = []

        for i, filename in enumerate(sorted(os.listdir(input_test_defects), key=self.custom_sort_key2)):
            print(f'Loading image: {i}, named: {filename}')
            
            # Loading the defect image
            if filename.endswith(".jpg") and i <= image_nb:
                defect_img = cv2.imread(f'{input_test_defects}/{filename}')
                img_name, _ = os.path.splitext(filename)
            
                # Read the information to detect the defect positions (possibly multiple defects)
                with open(f'{input_test_defects}/{img_name}.txt', 'r') as file:
                    contents = file.readlines()  # Read all lines to handle multiple defects
                    print(contents)
                    
                for content in contents:
                    data_list = content.split()
                    content_float = [float(i) for i in data_list]
                    lacalisation_defect.append(content_float)

                    # Create the matrix to know the exact defect position
                    # Start by creating a matrix full with 0s (this is done once)
                    zero_array_shape = list(defect_img.shape)
                    zero_array_shape[2] = 1
                    zero_channel = np.zeros(zero_array_shape)

                    # Find the pixel position for each defect in the image
                    x_center_norm = content_float[1]
                    y_center_norm = content_float[2]
                    width_defect_norm = content_float[3]
                    height_defect_norm = content_float[4]

                    # Set the width and height of the image
                    height_img, width_img, _ = defect_img.shape
                    
                    # Denormalize the position to detect the defect
                    x_center = x_center_norm * width_img
                    y_center = y_center_norm * height_img
                    width_defect = width_defect_norm * width_img
                    height_defect = height_defect_norm * height_img

                    # Calculate bounding box coordinates
                    x_min = int(x_center - width_defect / 2)
                    x_max = int(x_center + width_defect / 2)
                    y_min = int(y_center - height_defect / 2)
                    y_max = int(y_center + height_defect / 2)

                    # Ensure the coordinates are within image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width_img, x_max)
                    y_max = min(height_img, y_max)

                    # Add 1 to the defect axis where the defect is located
                    zero_channel[y_min:y_max, x_min:x_max] += 1  # Accumulate 1s for multiple defects

                # Concatenate the original image with the defect location
                defect_annotated = np.concatenate((defect_img, zero_channel), axis=-1)
                print(f'\n\nThe data shape with defect is: {defect_annotated.shape}\n\n')
                test_defects.append(defect_annotated)

        # **************************************************************************************************
        # ********************************************** TEST **********************************************
        # **************************************************************************************************
        if test:
            # Save the image with a red bounding box where is supposed to be located the defect. 
            defect_img_anotation = []
            for i, img in enumerate(test_defects):
                img_array = np.array(img)
                # Extract the first 3 channles (color channels)
                img = img_array[:, :, :3].astype(np.uint8)
                # Extract the forth channle (labbel channel)
                label_channel = img_array[:, :, 3]
                # Find the contours in the label channel where the value is 1 (Defect area)
                contours, _ = cv2.findContours((label_channel == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw a red rectangle around the defected defects
                for contour in contours:
                    # Get the bounding box for each contour
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw the rectangle on the image (in red, with thickness of 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    defect_img_anotation.append(img)

            # Save the image to analyze where is located the defect
            for i, img in enumerate(defect_img_anotation):
                img = np.array(img)
                path = f'{data_path}/{defect_type}/test_a_original_image_defect_location'
                if not os.path.exists(path):
                    os.makedirs(path)
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, img)

        # --------------------------------------------------------------------------------------------------------------------
        #                            2. SEGMENT THE IMAGES TO KEEP ONLY THE WELDING
        # --------------------------------------------------------------------------------------------------------------------
        segment_images_no_defects = []
        segment_images_defects = []
        images_bounding_boxes_no_defects = []
        images_bounding_boxes_defects = []
        test_no_defects = np.array(test_no_defects)
        test_defects = np.array(test_defects)

        for i, image in enumerate(test_no_defects):
            print(f'Analysing image {i+1}')
            # Segment the original image
            bounding_boxes = self.segment_welds_opencv(image[:, :, :3])
            # Add the bounding box to the original image
            image_bounding_boxes = self.draw_bounding_boxes(image[:, :, :3], bounding_boxes)
            print(f'The shape for the bounding box is: {image_bounding_boxes.shape}')
            images_bounding_boxes_no_defects.append(image_bounding_boxes)

            # Crop the original image
            cropped_images = self.crop_weld_sections(image, bounding_boxes)
            segment_images_no_defects.extend(cropped_images)

        print(f'There are: {len(segment_images_no_defects)} images.')
        
        for i, image in enumerate(test_defects):
            print(f'Analyzing image {i+1}')
            # Segment the original image
            bounding_boxes = self.segment_welds_opencv(image[:, :, :3])
            # Add the bounding box to the original image
            image_bounding_boxes = self.draw_bounding_boxes(image[:, :, :3], bounding_boxes)
            print(f'The shape for the bounding box is: {image_bounding_boxes.shape}')
            images_bounding_boxes_defects.append(image_bounding_boxes)

            # Crop the original image
            cropped_images = self.crop_weld_sections(image, bounding_boxes)
            segment_images_defects.extend(cropped_images)

        print(f'There are: {len(segment_images_defects)} images.')

        # **************************************************************************************************
        # ********************************************** TEST **********************************************
        # **************************************************************************************************
        if test:
            # Save the images to see what the bounding boxes looks like
            path = f'{data_path}/{no_defect_type}/b_bounding_boxes'
            if not os.path.exists(path):
                os.makedirs(path)
            for i, img in enumerate(images_bounding_boxes_no_defects):
                img = img[:, :, :3]
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, img)
            path = f'{data_path}/{defect_type}/b_bounding_boxes'
            if not os.path.exists(path):
                os.makedirs(path)
            for i, img in enumerate(images_bounding_boxes_defects):
                img = img[:, :, :3]
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, img)

            # Save the images to see if the cropping works
            path = f'{data_path}/{no_defect_type}/c_cropping'
            if not os.path.exists(path):
                os.makedirs(path)
            for i, img in enumerate(segment_images_no_defects):
                img = np.array(img)
                img = img[:, :, :3]
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, img)
            path = f'{data_path}/{defect_type}/c_cropping'
            if not os.path.exists(path):
                os.makedirs(path)
            for i, img in enumerate(segment_images_defects):
                img = np.array(img)
                img = img[:, :, :3]
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, img)

        # --------------------------------------------------------------------------------------------------------------------
        #                                        3. SUBDIVISE THE CROPPED IMAGES
        # --------------------------------------------------------------------------------------------------------------------
        sub_images_no_defects = []
        sub_images_defects = []

        print(f'Subdivising the no defect images')
        for images_training in segment_images_no_defects:
            if not np.all(images_training[:, :, :3] == 0):
                sub_images_no_defects.extend(self.subdivise(images_training))
        test_no_defects = sub_images_no_defects
        print(f'\n\nThere are: {len(test_no_defects)} subdivised images without any defect that are not all 0s (black pixels)\n\n')

        print(f'Subdivising the defect images')
        for images_validating in segment_images_defects:
            if not np.all(images_validating[:, :, :3] == 0):
                sub_images_defects.extend(self.subdivise(images_validating))
        test_defects = sub_images_defects
        print(f'\n\nThere are: {len(test_defects)} subdivised images with defect that are not all 0s (black pixels)\n\n')

        # **************************************************************************************************
        # ********************************************** TEST **********************************************
        # **************************************************************************************************
        if test:
            # Save the images to see if the cropping works
            for i, img in enumerate(test_no_defects):
                img = np.array(img)
                print(f'The image shape is: {img.shape}')
                img = img[:, :, :3]
                path = f'{data_path}/{no_defect_type}/d_subdivise'
                if not os.path.exists(path):
                    os.makedirs(path)
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, img)
            for i, img in enumerate(test_defects):
                img = np.array(img)
                img = img[:, :, :3]
                path = f'{data_path}/{defect_type}/d_subdivise'
                if not os.path.exists(path):
                    os.makedirs(path)
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, img)

        test_no_defects = np.array(test_no_defects)
        test_defects = np.array(test_defects)
        print(f'\n\nThe shape for test_no_defects is: {test_no_defects.shape} amd the shape for test_defects is: {test_defects.shape}\n\n')


        # --------------------------------------------------------------------------------------------------------------------
        #                               4. DELETE THE IMAGES THAT HAVE ALL PIXELS EQUALS TO 0
        # --------------------------------------------------------------------------------------------------------------------
        print(f'Deleting black images')
        filtered_no_defects = []
        cptr = 0
        threshold = 0.65 

        for i, testing_no_defects in enumerate(test_no_defects):
            # Flattening the first 3 channels (ignoring the defect channel)
            pixel_count = np.prod(testing_no_defects[:, :, :3].shape)
            # Count the number of zero pixels in the first 3 channels
            zero_count = np.sum(testing_no_defects[:, :, :3] == 0)
            # Calculate the percentage of zero pixels
            zero_percentage = zero_count / pixel_count
            # If less than threshold, keep the image
            if zero_percentage <= threshold:
                filtered_no_defects.append(testing_no_defects)
            else:
                cptr += 1

        print(f'We removed: {cptr} no defects images')
        print(f'There are: {len(filtered_no_defects)} images without defects after deleting black images')
                
        filtered_defects = []
        cptr = 0 

        for i, testing_defects in enumerate(test_defects):
            # Flattening the first 3 channels (ignoring the defect channel)
            pixel_count = np.prod(testing_defects[:, :, :3].shape)
            # Count the number of zero pixels in the first 3 channels
            zero_count = np.sum(testing_defects[:, :, :3] == 0)
            # Calculate the percentage of zero pixels
            zero_percentage = zero_count / pixel_count
            # If less than 80% of pixels are zeros, keep the image
            if zero_percentage <= threshold:
                filtered_defects.append(testing_defects)
            else:
                cptr += 1

        print(f'We removed: {cptr} defects images')
        print(f'There are: {len(filtered_defects)} images with defects after deleting black images')

        filtered_no_defects = np.array(filtered_no_defects)
        filtered_defects = np.array(filtered_defects)
        print(f'The shape for filtered_no_defects is: {filtered_no_defects.shape} and for filtered_defects is: {filtered_defects.shape}')

        # **************************************************************************************************
        # ********************************************** TEST **********************************************
        # **************************************************************************************************
        if test:
            # Save the images to see if the cropping works
            for i, img in enumerate(filtered_no_defects):
                img = img[:, :, :3]
                path = f'{data_path}/{no_defect_type}/e_delete_0s'
                if not os.path.exists(path):
                    os.makedirs(path)
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, img)
            for i, img in enumerate(filtered_defects):
                img = np.array(img)
                img = img[:, :, :3]
                path = f'{data_path}/{defect_type}/e_delete_0s'
                if not os.path.exists(path):
                    os.makedirs(path)
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, img)

            # Save the image with a red bounding box where is supposed to be located the defect. 
            defect_img_anotation = []
            for i, images in enumerate(filtered_defects):
                # Extract the first 3 channles (color channels)
                img = images[:, :, :3].astype(np.uint8)
                # Extract the forth channle (labbel channel)
                label_channel = images[:, :, 3]
                # Find the contours in the label channel where the value is 1 (Defect area)
                contours, _ = cv2.findContours((label_channel == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw a red rectangle around the defected defects
                for contour in contours:
                    # Get the bounding box for each contour
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw the rectangle on the image (in red, with thickness of 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    defect_img_anotation.append(img)

            # Save the image to analyze where is located the defect
            for i, images in enumerate(defect_img_anotation):
                images = np.array(images)
                path = f'{data_path}/{defect_type}/test_e_defect_location'
                if not os.path.exists(path):
                    os.makedirs(path)
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, images)

        # --------------------------------------------------------------------------------------------------------------------
        #                          5. FIX LABELS (IF GRAYSCALE CHANNELS = 0 [BLACK], PUT LABEL TO 0)
        # --------------------------------------------------------------------------------------------------------------------
        """
        There is a small innacuracy with the defect and the welding. Indeed some defect are going to be on 
        black pixels. It is important to remove them and put the label value to 0 there to know that there
        are no defects when the pixels are black.
        """
        print('Fixing the labels')

        # Doing it for the defect pieces
        # Check where all Grayscale values (channels 0, 1, 2) are zero
        gray_zero_mask = np.all(filtered_defects[:, :, :, 0:3] == 0, axis=-1)
        # Find where the label (channel 3) is 1
        label_one_mask = filtered_defects[:, :, :, 3] == 1

        condition = gray_zero_mask & label_one_mask

        # Modify the label to 0 where the condition is True
        filtered_defects[:, :, :, 3][condition] = 0

        # **************************************************************************************************
        # ********************************************** TEST **********************************************
        # **************************************************************************************************
        if test:
            # Put the new defect area in red
            visual_defects = filtered_defects[:, :, :, 0:3].copy()
            target_defects = filtered_defects[:, :, :, 3].copy()
            visual_defects[condition] = [255, 0, 0]

            # Save the image with a red bounding box where is supposed to be located the defect. 
            defect_img_anotation = []
            for i, (images, target) in enumerate(zip(visual_defects,target_defects)):
                # Extract the first 3 channles (color channels)
                img = images
                # Extract the forth channle (labbel channel)
                label_channel = target
                # Find the contours in the label channel where the value is 1 (Defect area)
                contours, _ = cv2.findContours((label_channel == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw a red rectangle around the defected defects
                for contour in contours:
                    # Get the bounding box for each contour
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw the rectangle on the image (in red, with thickness of 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    defect_img_anotation.append(img)

            path = f'{data_path}/{defect_type}/test_f_label_correction'
            print(f'The path is: {path}')
            if not os.path.exists(path):
                print(f'The path do not exist')
                os.makedirs(path)
            else:
                print('The path exists')

            # Save the image to analyze where is located the defect
            for i, images in enumerate(defect_img_anotation):
                images = np.array(images)
                img_path = os.path.join(path, f'weld_{i+1}.png')
                cv2.imwrite(img_path, images)

        # --------------------------------------------------------------------------------------------------------------------
        #                          5. NORMALIZE THE INPUT (VALUES BETWEEN 0-1)
        # --------------------------------------------------------------------------------------------------------------------
        print(f'Normalizing the data')
        train_no_defects_loss_norm = self.normalize_classification(filtered_no_defects, max_pixel_value)
        valid_defects_loss_norm = self.normalize_classification(filtered_defects, max_pixel_value)
        print(f'The shape after the normalization is: {train_no_defects_loss_norm.shape} without defects')
        print(f'The shape after the normalization is: {valid_defects_loss_norm.shape} with defects')
            
        return train_no_defects_loss_norm, valid_defects_loss_norm 



    def subdivise(self, image, overlap_percent=0.40):
        """
        Subdivise an input image with an overlap value.
        Inputs: - image: The input image that need to be subdivised
                - overlap_percent: The pourcentage that need to be overlapped in the image, .0 mean no overlap
        Output: - sub_images: The subdivised images. 
        """
        height, width = image.shape[:2]
        print(f"Original dimensions: ({width}, {height})")

        overlap_width = int(self.width * overlap_percent)
        overlap_height = int(self.height * overlap_percent)
        step_width = self.width - overlap_width
        step_height = self.height - overlap_height

        sub_images = []
        for i in range(0, width, step_width):
            for j in range(0, height, step_height):
                left = i
                right = min(i + self.width, width)
                top = j
                bottom = min(j + self.height, height)

                # Create a black canvas of size 256x256
                img_sub = np.zeros((self.height, self.width, image.shape[2]), dtype=image.dtype)
                
                # Calculate the size of the actual image content
                content_height = bottom - top
                content_width = right - left
                
                # Copy the image content onto the black canvas
                img_sub[:content_height, :content_width] = image[top:bottom, left:right]

                print(f'The shape of the sub image at ({i}, {j}) is: {img_sub.shape}')
                sub_images.append(img_sub)

        return sub_images



    def segment_welds_opencv(self, image, train=False):
        """
        Take an input image which is a welding piece with a black background. 
        Create a bounding box around the welding present in the original image.
        input - image: A gray scale image of the original welding piece (4 channels, 3 channels for color and 1 channel for label)
        output - bounding_boxes: An array representing the location of the bounding box: x, y, width, height
        """
        # # Put the image in unint8 format:
        if not train:
            image = (255 * image / np.max(image)).astype(np.uint8)
        # Convert the image to grayscale with only one channel
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the bounding boxes that are too small
        min_area = 60
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

        # Create bounding boxes for each contour 
        bounding_boxes = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        return bounding_boxes


    def draw_bounding_boxes(self, image, bounding_boxes, train=False):
        """
        Used only to test. 
        Visual representation of where the bounding boxes are located.
        Input- image: the original image
               bounding_boxes: The bounding box find by the algorithm
        output- img: An array of the original image with a green box representing where each box got found in the original image. 
        """
        # Convert the image in uint8:
        if not train:
            image = np.uint8(255 * image / np.max(image))
        # Draw each bounding box on the image
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box with 2px thickness

        return image


    def crop_weld_sections(self, image, bounding_boxes):
        """
        Crop the image from the bounding boxes where the welding piece are. 
        input - Image: the original image
            - bounding_boxes: The bounding boxes where the welding piece are
        output - cropped_image: A cropped image with the welding piece
        """
        cropped_images = []
        for box in bounding_boxes:
            x, y, w, h = box
            cropped_image = image[y:y+h, x:x+w]
            cropped_images.append(cropped_image)
        
        print(f'There are : {len(cropped_images)} elements in the welding piece')

        return cropped_images
    

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
        

    def custom_sort_key2(self, filename):
        # Adjust the regular expression to ignore the file extension
        match = re.match(r'(\d+)_(\d+)_(\d+)_.*', filename)
        if match:
            # Return a tuple with three integer keys for sorting
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)), filename)
        else:
            # Handle filenames that do not match the expected format
            return (float('inf'), float('inf'), float('inf'), filename)