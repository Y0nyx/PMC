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

from sklearn.metrics import mean_absolute_error

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

def createPredImg(input_train, input_test, difference_reshaped, image, num, path_results):
    fig, axes = plt.subplots(1, 3, figsize=(24,8))
    fig.suptitle('Input data vs Results')

    axes[0].set_title('Inputs')
    im0 = axes[0].imshow(input_train, cmap='gray')
    axes[1].set_title('Results')
    im1 = axes[1].imshow(input_test, cmap='gray')
    axes[2].set_title('difference')
    im2 = axes[2].imshow(difference_reshaped, cmap='jet')

    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    dir = f'{path_results}/image/result_model_{num+1}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}/{image+1}.png')
    plt.close()

def write_error_csv(mae, mae_threshold, error, path_results, image_number, l, k):
    print(f'The l value is: {l}')
    fieldnames = ['image_number', 'welding_number', 'subdivise_number', 'error', 'MAE_value', 'MAE_threshold']
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
            'image_number': image_number,
            'welding_number': l + 1,
            'subdivise_number': k + 1,
            'error': error,
            'MAE_value': mae,
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

    test = True
    if not test:
        data_processing = dp.DataProcessing(sub_width, sub_height)
        train_input, train_input_loss, valid_input, valid_input_loss, test_input = data_processing.get_data_processing_standard(data_path, max_pixel_value, test=False) #Put test to True later...  , defaut_images_norm

        _, row, column, channels = train_input.shape
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

            input_test_norm = test_input[0:num_train_regenerate]

            result_norm = prediction(build_model, input_test_norm)

            input_test_denorm, result_denorm, difference_reshaped = data_processing.de_normalize(input_test_norm, result_norm, max_pixel_value)

            for i in range(num_train_regenerate):
                createPredImg(input_test_denorm[i], result_denorm[i], difference_reshaped[i], i, j, path_results)

        print('The testing of the Neural Network has been done corectly!')

    #If we want to see if there is a welding error in a piece. 
    else:
        #Define a threshold
        mae_threshold = 0.10

        #Data processing and split the data into set (Train, test, valid).
        data_processing = dp.DataProcessing(sub_width, sub_height)
        test_input, defaut_images = data_processing.get_data_processing_blackout(data_path, max_pixel_value, test=True)
        print('data processing done. ')

        #Concatenate defect with non defect test images. 
        test_images = np.concatenate((defaut_images, test_input), axis=0)
        print(f'The shape of the defaut {defaut_images.shape}')
        print(f'The shape of the test_input {test_input.shape}')
        print(f'{test_images.shape}')
        del defaut_images
        del test_input

        _, row, column, channels = test_images.shape
        image_dimentions = (row, column, channels)

        data_frame = pd.read_csv(f'{path_results}/hp_search_results.csv')
        #List all the hp used during training. 
        learning_rate = data_frame['lr']
        #Selected model
        j = 1

        image_dimentions = (256, 256, 3) #TODO eventually change the hardcoded value
        #Create the model that will be used
        model = mod.AeModels(float(learning_rate.iloc[j]), monitor_loss, monitor_metric, image_dimentions)
        build_model = model.aes_defect_detection()
        name = f"model{j+1}"
        build_model.load_weights(f'{filepath_weights}/search_{name}')

        for i in range(test_images.shape[0]):
            #Segmente the welding image
            image_number = (i // 4) + 1
            print(f'Currently analysing image number: {image_number}')
            test_images[i] = cv2.cvtColor(test_images[i], cv2.COLOR_RGB2BGR) 
            welding_segmented = data_processing.segment(test_images[i])

            #Subdivise the segmented image (For the number of welding that where in the image).
            for l, welding in enumerate(welding_segmented):
                print(f'The length of the segmented welding is: {len(welding_segmented)}')
                print(f'The new l value is: {l}')
                divised_welding_segmented = data_processing.subdivise(welding)
                #Do a prediction on every subdivised image
                for k, image in enumerate(divised_welding_segmented):
                    #Put the data into RGB format
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                    #cv2.imwrite(f'/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/result_paths/image_{k}.jpg', image)

                    #Normalize the data
                    image_normalize = data_processing.normalize(image, max_pixel_value)
                    image_normalize_expand = np.expand_dims(image_normalize, axis=0)

                    #Predict with the network
                    result_norm = prediction(build_model, image_normalize_expand)
                    #Evaluate the model
                    test_loss, test_metric = build_model.evaluate(image_normalize_expand, image_normalize_expand)
                    print(f'The test_loss is {test_loss}')

                    #Save the predicted image
                    #TODO save the predicted image, original image and error between the image. 

                    #Calculate the mae between the input and the prediction
                    result_norm = np.squeeze(result_norm)
                    image_normalize_flattened = image_normalize.flatten()
                    result_norm_flattened = result_norm.flatten()
                    mae = mean_absolute_error_hand(image_normalize, result_norm)
                    #mae = mean_absolute_error(image_normalize_flattened, result_norm_flattened)
                    print(f'The mae value is {mae}')

                    #Validate with the threshold
                    if mae >= mae_threshold:
                        error = 1
                    else:
                        error = 0

                    #Write to a csv file if there is a welding error. 
                    print('Writting info to csv...')
                    write_error_csv(mae, mae_threshold, error, path_results, image_number, l, k)


        #Take the final decision if there is a welding error. 
        