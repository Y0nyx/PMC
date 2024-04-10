#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 02/12/2024
Description: Used to *TEST the generation of patches. Will not be use when the right defauts detection 
             are done. In oder words this script is *TEMPORARY. 
"""

from keras.preprocessing import image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import data_processing as dp

#Add the path where you are doing your test. 
data_path = '/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/4k_images_blackout/image/segmented/image1'
IMG_SIZE = (256, 256) #row, column 
img_size_row = IMG_SIZE[0] #The size of the tested images. 
img_size_col = IMG_SIZE[1] #The size of the tested images. 
de_norm_value = 255 #Denormalization value. 
images = []

#Data loading
for i, filename in enumerate(os.listdir(data_path)):
    print(f'Loading image {i}')
    if filename.endswith(".jpg") and i < 10:
        img = cv2.imread(f'{data_path}/{filename}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    else:
        break
print('Loading done...')

#Apply the stain random noise to the original data
data_process = dp.DataProcessing(256, 256)
images_stain = []
for i, img in enumerate(images):
    print(f'Analysing image {i}')
    if i < 10:
        img = data_process.normalize(img, de_norm_value)
        images_stain.append(data_process.add_stain(img, 255))
    else:
        break
print('Stain added.')

#Apply the random blackout noise to the original data
data_process = dp.DataProcessing(256, 256)
images_blackout = []
for i, img in enumerate(images):
    print(f'Analysing image {i}')
    if i < 10:
        img = data_process.normalize(img, de_norm_value)
        print(img.shape)
        images_blackout.append(data_process.apply_random_blackout(img))
    else:
        break
print('Blackout added.')

for i, (img, img_stain) in enumerate(zip(images, images_blackout)):
    print(f'The original data is of type: {img.dtype}')
    if i < 10:
        #Denormalize data
        # img_stain = (img_stain * de_norm_value).astype('uint8')
        # img = (img * de_norm_value).astype('uint8')
        print(f'The modified data is of type: {img_stain.dtype}')

        fig, axes = plt.subplots(1,2)
        fig.suptitle('Donnee originale Vs Donnee Blackout')

        axes[0].set_title('Donnee origine')
        axes[0].imshow(img)
        axes[1].set_title('Donnee bruitee')
        axes[1].imshow(img_stain)

        plt.show(block=True)
