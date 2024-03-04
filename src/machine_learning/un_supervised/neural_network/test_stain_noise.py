#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 02/12/2024
Description: Used to *TEST the generation of patches. Will not be use when the right defauts detection 
             are done. In oder words this script is *TEMPORARY. 
"""

from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

import data_processing as dp

#Add the path where you are doing your test. 
data_path = '/home/jean-sebastien/Documents/s7/PMC/Data/images_cam_123/sub_images'
img_size_row = 256 #The size of the tested images. 
img_size_col = 256 #The size of the tested images. 
de_norm_value = 255 #Denormalization value. 
images = []

for filename in os.listdir(data_path):
    if filename.endswith(".png"):
        img = image.load_img(f'{data_path}/{filename}', target_size=(img_size_row, img_size_col))
        images.append(image.img_to_array(img))
images = np.array(images)

#Apply the stain random noise to the original data
data_process = dp.DataProcessing()
images_stain = []
for img in images:
    images_stain.append(data_process.add_stain(img))

for i, (img, img_stain) in enumerate(zip(images, images_stain)):
    print(f'The original data is of type: {img.dtype}')
    if i < 3:
        #Denormalize data
        img_stain = (img_stain * de_norm_value).astype('uint8')
        img = (img * de_norm_value).astype('uint8')
        print(f'The modified data is of type: {img_stain.dtype}')

        fig, axes = plt.subplots(1,2)
        fig.suptitle('Original data Vs Stain data')

        axes[0].set_title('Original data')
        axes[0].imshow(img)
        axes[1].set_title('Noisy data')
        axes[1].imshow(img_stain)

        plt.show(block=True)
