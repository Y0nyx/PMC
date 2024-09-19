#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 02/03/2024
Description: Testing the Neural Network. 
"""

import argparse
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import data_processing as dp
import model as mod

def evaluate_model(model, input_test, batch_size=32, verbose=1):
    test_loss = model.evaluate(
        x=input_test,
        y=input_test,
        batch_size=batch_size,
        verbose=verbose
    )
    print(f'Test loss: {test_loss:.3f}')

    return test_loss

def prediction(model, input_test):
    return model.predict(
        x=input_test,
        batch_size=None,
        verbose=1,
        steps=None,
        callbacks=None
    )

def createPredImg(input_train, input_test, difference_reshaped, image, num, path_results, title, type):
    fig, axes = plt.subplots(1, 3, figsize=(24,8))
    fig.suptitle(title)

    vmin = min(input_train.min(), input_test.min())
    vmax = max(input_train.max(), input_test.max())

    axes[0].set_title('Inputs')
    im0 = axes[0].imshow(input_train, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Results')
    im1 = axes[1].imshow(input_test, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title('difference')
    im2 = axes[2].imshow(difference_reshaped, cmap='jet')

    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    dir = f'{path_results}/image/result_{type}_{num+1}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}/{image+1}.png')
    plt.close()

def createPredImg2(input_train, input_test, difference_reshaped, path_results):
    fig, axes = plt.subplots(1, 3, figsize=(24,8))
    fig.suptitle('Input data vs Results')

    axes[0].set_title('Donnee entree')
    im0 = axes[0].imshow(input_train, cmap='gray')
    axes[1].set_title('Resultats')
    im1 = axes[1].imshow(input_test, cmap='gray')
    axes[2].set_title('difference')
    im2 = axes[2].imshow(difference_reshaped, cmap='jet')

    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.savefig(path_results)
    plt.close()

def write_error_csv(test_loss, psnr_value, ssim_value, mae_threshold, error, path_results, image, view, weld_num, i):
    """
    Write the information to a csv file to know which piece has a welding error. 
    """
    fieldnames = ['image_number', 'view_number', 'segmented_number', 'sub_num', 'error', 'MAE_value', 'psnr_value', 'ssim_value', 'MAE_threshold']
    csv_path = f'{path_results}/prediction/prediction_results.csv'

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Determine if the file is new or already exists
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file is new/empty
        if not file_exists:
            writer.writeheader()

        row_dict = {
            'image_number': image,
            'view_number': view + 1,
            'segmented_number': weld_num + 1,
            'sub_num': i+1,
            'error': error,
            'MAE_value': test_loss,
            'psnr_value': psnr_value,
            'ssim_value': ssim_value,
            'MAE_threshold': mae_threshold,
        }
        writer.writerow(row_dict)

def mean_absolute_error_hand(image_normalize, result_norm):
    """
    Calculate the mae between two values. 
    """
    # Calculate the absolute differences
    absolute_differences = np.abs(result_norm - image_normalize)

    # Since we're dealing with single images, we can directly compute the sum of absolute differences
    # across all pixels and channels
    sum_absolute_differences = np.sum(absolute_differences)

    # The total number of elements (pixels * channels) in one image
    total_elements = np.prod(image_normalize.shape)

    # Calculate the Mean Absolute Error (MAE) by dividing the sum of absolute differences
    # by the total number of elements
    mae = sum_absolute_differences / total_elements

    return mae

def get_segmented_images(test_images, path_results):
    """
    Segment the image to only keep the welding of each input images. 
    """
    segmented_images = []
    image_view = 1
    for i, img in enumerate(test_images):
        image_number = (i // 4) + 1
        welding_segmented = data_processing.segment(img)
        
        result_path = f'{path_results}/image/segmented/image{image_number}'
        os.makedirs(result_path, exist_ok=True)

        for j, welding in enumerate(welding_segmented):
            cv2.imwrite(f'{result_path}/welding_segmented_{image_view}_segment_{j}_.jpg', welding)
        
        segmented_images.extend(welding_segmented)
        
        image_view += 1
        if image_view == 5:
            image_view = 1

    return segmented_images

def get_subdivise_images():
    """
    Subdivise the images in smaller images 
    """
    initial_data_path = '/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/4k_images_blackout/image/segmented'
    file_names = os.listdir(initial_data_path)

    for name in file_names:
        data_path = f'/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/4k_images_blackout/image/segmented/{name}'
        images = []
        sorted_filenames = sorted(os.listdir(data_path), key=custom_sort_key)
        #Data loading
        for filename in sorted_filenames:
            if filename.endswith(".jpg"):
                images.append(cv2.imread(f'{data_path}/{filename}'))

        for i, img in enumerate(images):
            parts = sorted_filenames[i].split('_')
            result_path = f'{path_results}/image/subdivised/{name}/imageView_{parts[2]}/segment_{parts[4]}'
            os.makedirs(result_path, exist_ok=True)

            img_subdivise = data_processing.subdivise(img)
            for i, small_img in enumerate(img_subdivise, start=1):
                cv2.imwrite(f'{result_path}/welding_subdivised_{i}.jpg', small_img)

def custom_sort_key(filename):
    match = re.match(r'(\d+)([a-z]+)_.*', filename)
    if match:
        # Return a tuple with the numeric part as an integer and the prefix
        return (int(match.group(1)), match.group(2))
    else:
        # If no match, return a tuple that puts the filename last
        return (float('inf'), filename)  
    
def custom_sort_key2(filename):
    match = re.search(r'_([0-9]+)\.jpg$', filename)
    if match:
        # Return the numeric part as an integer for sorting
        return int(match.group(1))
    else:
        # If no match, return a large number to put the filename last
        return float('inf')
    
def custom_sort_key3(filename):
    match = re.search(r'image(\d+)', filename, re.IGNORECASE)
    if match:
        # Return the numeric part as an integer for sorting
        return int(match.group(1))
    else:
        # If no match, return a large number to put the filename last
        return float('inf')
    
def custom_sort_key4(filename):
    match = re.search(r'imageView_(\d+)', filename, re.IGNORECASE)
    if match:
        # Return the numeric part as an integer for sorting
        return int(match.group(1))
    else:
        # If no match, return a large number to put the filename last
        return float('inf')

def custom_sort_key5(filename):
    match = re.search(r'segment_(\d+)', filename, re.IGNORECASE)
    if match:
        # Return the numeric part as an integer for sorting
        return int(match.group(1))
    else:
        # If no match, return a large number to put the filename last
        return float('inf')

def argparser():
    parser = argparse.ArgumentParser(description='Argument used in the code passed by the bash file.')

    parser.add_argument('--PATH_RESULTS', type=str, default='/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/B_First_HP_Search',
                        help='path where the results are going to be stored at.')
    parser.add_argument('--NBEST', type=int, default=10,
                        help='Number of best hp search that will be taken to train the model with.')
    parser.add_argument('--NUM_TRAIN_REGENERATE', type=int, default=20,
                        help='Number of test data that will be used for the generation.')
    parser.add_argument('--MONITOR_LOSS', type=str, default='mean_absolute_error',
                         help='The metric being used for the loss function.')
    parser.add_argument('--MONITOR_METRIC', type=str, default ='mean_squared_error', 
                        help='The metric that is being monitored.')
    parser.add_argument('--FILEPATH_WEIGHTS', type=str, default='/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/B_First_HP_Search/training_weights/',
                        help='Path where are stored the weights.')
    parser.add_argument('--DATA_PATH', type=str, default='/home/jean-sebastien/Documents/s7/PMC/Data/images_cam_123/sub_images',
                        help='Path where are located the data.')
    parser.add_argument('--MAX_PIXEL_VALUE', type=int, default=255,
                        help='Maximum pixel value for the analysed original image.')
    parser.add_argument('--SUB_WIDTH', type=int, default=256,
                        help='Width of the image after the subtitution of the images')
    parser.add_argument('--SUB_HEIGHT', type=int, default=256,
                        help='Width of the image after the subtitution of the images')

    return parser.parse_args()

if __name__ =='__main__':
    args = argparser()

    path_results = args.PATH_RESULTS
    nbest = args.NBEST
    num_train_regenerate = args.NUM_TRAIN_REGENERATE
    monitor_loss = args.MONITOR_LOSS
    monitor_metric = args.MONITOR_METRIC
    filepath_weights = args.FILEPATH_WEIGHTS
    data_path = args.DATA_PATH
    max_pixel_value = args.MAX_PIXEL_VALUE
    sub_width = args.SUB_WIDTH
    sub_height = args.SUB_HEIGHT

    test = False
    if not test:
        data_processing = dp.DataProcessing(sub_width, sub_height)
        test_input_no_defects, test_input_defects = data_processing.get_data_processing_stain_PMC860_test(data_path, max_pixel_value) #Put test to True later...  , defaut_images_norm

        _, row, column, channels = test_input_no_defects.shape
        image_dimentions = (row, column, channels)
        print(image_dimentions)

        data_frame = pd.read_csv(f'{path_results}/hp_search_results.csv')
        #List all the hp used during training. 
        learning_rate = data_frame['lr']

        for j in range(nbest):
            model = mod.AeModels(float(learning_rate.iloc[j]), monitor_loss, monitor_metric, image_dimentions)
            build_model = model.aes_defect_detection()   #Change this line if the model change. 

            name = f"model{j+1}"
            build_model.load_weights(f'{filepath_weights}/search_{name}')

            test_no_defects_norm = test_input_no_defects[0:num_train_regenerate]
            test_defects_norm = test_input_defects[0:num_train_regenerate]

            print(f'Shape test_no_defects_norm: {test_no_defects_norm.shape}, shape test_defects_norm: {test_defects_norm.shape}')
            result_no_defects_norm = prediction(build_model, test_no_defects_norm)
            result_defects_norm = prediction(build_model, test_defects_norm)

            test_no_defects_denorm, result_no_defects_denorm, difference_no_defects_reshaped = data_processing.de_normalize(test_no_defects_norm, result_no_defects_norm, max_pixel_value)
            test_defects_denorm, result_defects_denorm, difference_defects_reshaped = data_processing.de_normalize(test_defects_norm, result_defects_norm, max_pixel_value)


            mse_value_no_defects = []
            mse_value_defects = []

            for i in range(num_train_regenerate):
                print(f'Regenarating image: {i+1}')
                createPredImg(test_no_defects_denorm[i], result_no_defects_denorm[i], difference_no_defects_reshaped[i], i, j, path_results, 'Input data vs Results for no defects', 'no_defects')
                createPredImg(test_defects_denorm[i], result_defects_denorm[i], difference_defects_reshaped[i], i, j, path_results, 'Input data vs Results for defects', 'defects')

                # Calculating the metrics
                mask = test_no_defects_denorm[i] != 0
                masked_test = test_no_defects_denorm[i][mask]
                masked_result = result_no_defects_denorm[i][mask]

                mse_value_no_defects.append(mean_squared_error(masked_test, masked_result))

                mask = test_defects_denorm[i] != 0
                masked_test = test_defects_denorm[i][mask]
                masked_result = result_defects_denorm[i][mask]

                mse_value_defects.append(mean_squared_error(masked_test, masked_result))

            
            np.array(mse_value_no_defects)
            np.array(mse_value_defects)

            mean_mse_no_defect = np.mean(mse_value_no_defects)
            mean_mse_defect = np.mean(mse_value_defects)

            print(f'The mse over the test set for no defect is: {mean_mse_no_defect} and for defect is: {mean_mse_defect}')



        print('The testing of the Neural Network has been done corectly!')

    #If we want to see if there is a welding error in a piece. 
    else:
        #Define a threshold
        ssim_treshold = 0.80
        direct = '/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/4k_images_stain_01/image/subdivised'

        #Data processing and split the data into set (Train, test, valid).
        data_processing = dp.DataProcessing(sub_width, sub_height)
        test_input, defaut_images = data_processing.get_data_processing_stain(data_path, max_pixel_value, test=True)

        #Concatenate defect with non defect test images. 
        test_images = np.concatenate((defaut_images, test_input), axis=0)
        print(f'The shape of the defaut {defaut_images.shape}')
        print(f'The shape of the test_input {test_input.shape}')
        print(f'{test_images.shape}')
        del defaut_images
        del test_input

        #Get the hp values
        data_frame = pd.read_csv(f'{path_results}/hp_search_results.csv')
        #List all the hp used during training. 
        learning_rate = data_frame['lr']
        #Selected model
        j = 0

        #Create the model that will be used
        image_dimentions = (256, 256, 3) #TODO eventually change the hardcoded value
        model = mod.AeModels(float(learning_rate.iloc[j]), monitor_loss, monitor_metric, image_dimentions)
        build_model = model.aes_defect_detection()
        name = f"model{j+1}"
        build_model.load_weights(f'{filepath_weights}/search_{name}')

        do_data_processing = False
        if do_data_processing: 
            #Get the segmented images.
            segmented_images = get_segmented_images(test_images, path_results)
            #Get the subdivised images. 
            get_subdivise_images()

        #Main loop to predict if there is a defect or not
        sorted_folder = sorted(os.listdir(direct), key=custom_sort_key3)

        for image, folder in enumerate(sorted_folder):
            print(f'The folder is {folder}')

            camera_directory = f'{direct}/{folder}'
            sorted_folder_cam = sorted(os.listdir(camera_directory), key=custom_sort_key4)

            for view, camera_view in enumerate(sorted_folder_cam):
                print(f'The camera_view is {camera_view}')

                welding_dir = f'{direct}/{folder}/{camera_view}'
                sorted_folder_welding = sorted(os.listdir(welding_dir), key=custom_sort_key5)

                for weld_num, welding in enumerate(sorted_folder_welding):
                    print(f'The camera_view is {welding}')

                    final_path = f'{direct}/{folder}/{camera_view}/{welding}'
                    sorted_filenames = sorted(os.listdir(final_path), key=custom_sort_key2)
            
                    #Data loading
                    images = []

                    for filename in sorted_filenames:
                        if filename.endswith(".jpg"):
                            img = cv2.imread(f'{final_path}/{filename}')
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)

                    for i, img in enumerate(images):
                        #Normalize the input data
                        image_normalize = data_processing.normalize(img, max_pixel_value)
                        image_normalize_expand = np.expand_dims(image_normalize, axis=0)
                        #Predict on the trained model
                        result_norm = prediction(build_model, image_normalize_expand)
                        #Evaluate the performances of the model
                        test_loss, test_metric = build_model.evaluate(result_norm, result_norm)

                        #Denormalize the data 
                        input_test_denorm, result_test_denorm, difference_reshaped_denorm = data_processing.de_normalize(image_normalize_expand, result_norm, max_pixel_value)
                        result_test_denorm = result_test_denorm.squeeze()
                        difference_reshaped_denorm = difference_reshaped_denorm.squeeze()

                        #Save the predicted image
                        path_results = f'{final_path}/prediction/'
                        os.makedirs(path_results, exist_ok=True)
                        new_path_results = f'{path_results}/{sorted_filenames[i]}'

                        createPredImg2(img, result_test_denorm, difference_reshaped_denorm, new_path_results)

                        #Calculate error image metrics
                        print(f'The shape of the image is: {img.shape}')
                        psnr_value = psnr(img, result_test_denorm, data_range=255)
                        ssim_value = ssim(img, result_test_denorm, data_range=255, channel_axis=-1)
                        print(f'The psnr value is {psnr_value}')
                        print(f'Tje ssim value is {ssim_value}')

                        #Taking decision if defect or not
                        if ssim_value <= ssim_treshold:
                            error = 1
                        else:
                            error = 0

                        #Write the information to a csv file
                        write_error_csv(test_loss, psnr_value, ssim_value, ssim_treshold, error, direct, image, view, weld_num, i)
