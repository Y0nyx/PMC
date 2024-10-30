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

from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, precision_score

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

def calculate_pixelwise_mse(predicted_img, target_img):
    """
    Calculate the mse value between the prediction and the target image pixel wise
    Inputs - predicted_img: The output of the neural network of dimension sample x row x column x channles
           - target_img:    The target of the neural network of dimension sample x row x column x channles
    Output - pixel_wise_mse: The pixel wise mse between the prediction and the target of dimension sample x row x column x channles
    """
    assert predicted_img.shape == target_img.shape, "The prediction and the target are not the same shape which mean there is an issue with these values in the code"

    # Calculate the pixel-wise MSE
    mse_map = np.square(predicted_img - target_img)

    return mse_map


def calculate_pixelwise_rmse(predicted_img, target_img):
    """
    #TODO Maybe rename this to absolute error, because we are not doing any mean here?
    Calculate the mse value between the prediction and the target image pixel wise
    Inputs - predicted_img: The output of the neural network of dimension sample x row x column x channles
           - target_img:    The target of the neural network of dimension sample x row x column x channles
    Output - pixel_wise_mae: The pixel wise rmse between the prediction and the target of dimension sample x row x column x channles
    """
    assert predicted_img.shape == target_img.shape, "The prediction and the target are not the same shape which mean there is an issue with these values in the code"

    # Calculate the pixel-wise rmse
    rmse_map = np.sqrt(np.square(predicted_img - target_img))

    return rmse_map


def calculate_pixelwise_mae(predicted_img, target_img):
    """
    #TODO Maybe rename this to absolute error, because we are not doing any mean here?
    Calculate the mse value between the prediction and the target image pixel wise
    Inputs - predicted_img: The output of the neural network of dimension sample x row x column x channles
           - target_img:    The target of the neural network of dimension sample x row x column x channles
    Output - pixel_wise_mae: The pixel wise mae between the prediction and the target of dimension sample x row x column x channles
    """
    assert predicted_img.shape == target_img.shape, "The prediction and the target are not the same shape which mean there is an issue with these values in the code"

    # Calculate the pixel-wise MAE
    mae_map = np.abs(predicted_img - target_img)

    return mae_map


def classify_pixels_with_confidence(Map, thr, scale_factor=10):
    """
    Inputs:
    - Map: The error map for each pixel of the prediction (sample x row x column x channels).
    - thr: The threshold value for the MAE to consider if there is an error in the welding piece or not.
    - scale_factor: A factor to control how sharply the confidence transitions from 0 to 1. Higher values make the confidence more binary, lower values make it smoother.
    
    Output:
    - confidence_map: A map where values between 0 and 1 represent the confidence of an error in each pixel.
    """
    # Calculate confidence as a function of how much the error exceeds the threshold
    # Use a sigmoid transformation to get smooth values between 0 and 1
    confidence_map = 1 / (1 + np.exp(-scale_factor * (Map - thr)))

    return confidence_map


def binary_classification_analysis(binary_map, target):
    """
    Calculate the number of true possitif, false possitif, true negatif, false negatif for every pixel in every predicted images
    Inputs - binary_map: A binary map where 0 represent no error in the pixel (value smaller than the threshold)
                          and 1 represent an error in the pixel (value bigger than the threshold)  Dimensions: sample x row x column x channles
    Output - true_possirif  : Number of True possitif predictions
             false_possitif : Number of False possitifs predictions
             true_negatif   : Number of True Negatif predictions
             false_negatif  : Number of False Negatif predictions
    """
    # There are three channels in the binary map associate with the same target. 
    target_replicated = np.repeat(target[:, :, :, np.newaxis], binary_map.shape[3], axis=3)
    # Flatten the data to allow simple operation to do the comparaison
    binary_map_fn = binary_map.flatten()
    target_fn = target_replicated.flatten()

    tp = np.sum((binary_map_fn == 1) & (target_fn == 1))
    fp = np.sum((binary_map_fn == 1) & (target_fn == 0))
    tn = np.sum((binary_map_fn == 0) & (target_fn == 0))
    fn = np.sum((binary_map_fn == 0) & (target_fn == 1))

    return tp, fp, tn, fn


def precision(tp, fp):
    """
    Calculate the precision of the predictions
    Inputs : tp  : Number of True possitif predictions
             fp : Number of False possitifs predictions
    Output : precision : The precision of the neural network
    """
    if tp + fp == 0:
        return 0
    precision = tp / (tp + fp)

    return precision


def recall(tp, fn):
    """
    Calculate the recall of the predictions
    Inputs : tp  : Number of True possitif predictions
             fn  : Number of False Negatif predictions
    Output : recall : The recall of the neural network
    """
    if tp + fn == 0:
        return 0
    recall = tp / (tp+fn)

    return recall 


def accuracy(tp, fp, tn, fn):
    """
    Calculate the accuracy of the predictions
    Inputs - tp  : Number of True possitif predictions
             fp : Number of False possitifs predictions
             tn   : Number of True Negatif predictions
             fn  : Number of False Negatif predictions
    Output - accuracy: The accuracy of the neural network 
    """
    if tp + fp + tn + fn == 0:
        return 0
    accuracy = (tp + tn) / (tp+fp+tn+fn)

    return accuracy


def f1score(precision, recall):
    """
    Calculate the f1 score of the predictions
    Inputs - precision : The precision of the neural network 
             recall : The recall of the neural network
    Output - f1score : The F1 score
    """
    if precision+recall == 0:
        return 0
    f1_score = 2* ((precision*recall) / (precision+recall))

    return f1_score

def find_optimal_threshold_and_plot_roc(probabilities, target, path_results):
    """
    Find the optimal threshold based on the ROC curve and plot the ROC curve.
    
    Inputs:
    - probabilities: The output probabilities from the model (sample x row x column x channels).
    - target: The ground truth binary values (sample x row x column), 1 channel.
    
    Outputs:
    - optimal_threshold: The threshold that gives the best trade-off between TPR and FPR.
    """
    # Ensure target is binary (0 or 1)
    assert np.array_equal(np.unique(target), [0, 1]), "Target must be binary (0 or 1)."

    # Option 1: Average probabilities across channels to reduce to single channel
    probabilities_avg = np.mean(probabilities, axis=-1)  # Shape: (sample, row, column)

    # Option 2: Alternatively, you can use max probabilities across channels
    # probabilities_avg = np.max(probabilities, axis=-1)  # Shape: (sample, row, column)

    # Flatten both the averaged probabilities and the target
    probabilities_flat = probabilities_avg.flatten()
    target_flat = target.flatten()

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(target_flat, probabilities_flat)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Create directory if it doesn't exist and save the plot
    dir = f'{path_results}/image'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}/result_roc_curve.png')
    
    # Find the threshold closest to the top-left corner (maximizing TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold


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
    
def box_plot(metrics, name_metric):
    plt.figure(figsize=(18, 12))
    plt.boxplot(metrics, labels=['no_defects', 'defects'])
    plt.title(f'Test set performances for the {name_metric} for non defect and defect images, without considering the defect location.')
    plt.ylabel('Error value')
    dir1 = f'{path_results}/image/boxPlot_metrics'
    if not os.path.exists(dir1):
        os.makedirs(dir1)
    plt.savefig(f'{dir1}/boxpliot_{name_metric}.png')
    plt.close()

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
    parser.add_argument('--TEST', action='store_true', default=True,
                        help='If the user want to do test ans save images.')

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
    test = args.TEST

    regression = False
    classification = True

    if regression:
        data_processing = dp.DataProcessing(sub_width, sub_height)
        test_input_no_defects, test_input_defects = data_processing.get_data_processing_stain_PMC860_test(data_path, max_pixel_value)

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
    elif classification:
        # Loading the non defect and defect test data. 
        data_processing = dp.DataProcessing(sub_width, sub_height)
        test_input_no_defects, test_input_defects = data_processing.get_data_processing_stain_PMC860_test_classification(data_path, max_pixel_value, test)
        
        # Creating the deep learning model architecture
        _, row, column, channels = test_input_no_defects.shape
        image_dimentions = (row, column, channels-1) # Excluding the last channel, because it represent the label (i.e. defect or not)
        # Selecting the best HPs and the selected model
        data_frame = pd.read_csv(f'{path_results}/hp_search_results.csv')
        learning_rate = data_frame['lr']
        model = mod.AeModels(float(learning_rate.iloc[0]), monitor_loss, monitor_metric, image_dimentions)
        build_model = model.aes_defect_detection()   #Change this line if the model change. 
        # Building the selected model. 
        name = f"model1"
        build_model.load_weights(f'{filepath_weights}/search_{name}')

        # Doing inference with the network (Regenerating the image from the input, i.e. regression)
        result_no_defects_norm = prediction(build_model, test_input_no_defects[:, :, :, :3]) # Excluding channel 4 because it is the label. 
        result_defects_norm = prediction(build_model, test_input_defects[:, :, :, :3])
        # Creating a label for these predictions
        no_defect_label = test_input_no_defects[:, :, :, 3]
        defect_label = test_input_defects[:, :, :, 3]

        # Un-standardizing the data, to get the original input data format. 
        test_no_defects_denorm, result_no_defects_denorm, difference_no_defects_reshaped = data_processing.de_normalize(test_input_no_defects[:, :, :, :3], result_no_defects_norm, max_pixel_value)
        test_defects_denorm, result_defects_denorm, difference_defects_reshaped = data_processing.de_normalize(test_input_defects[:, :, :, :3], result_defects_norm, max_pixel_value)

        # Putting the no_defect and defect data together #TODO do this operation at the beginning of the code. 
        test_data = np.concatenate((test_no_defects_denorm, test_defects_denorm), axis=0)
        test_target_regression = np.concatenate((result_no_defects_denorm, result_defects_denorm), axis=0)
        test_target_classification = np.concatenate((no_defect_label, defect_label), axis=0)
        
        # Calculating the MAE, MSE and RMSE a for each pixel for the prediciton
        test_maeMap = calculate_pixelwise_mae(test_data, test_target_regression)
        test_mseMap = calculate_pixelwise_mse(test_data, test_target_regression)
        test_rmseMap = calculate_pixelwise_rmse(test_data, test_target_regression)

        # Defining a threshold for every metrics to define an error in the welding piece
        mae_thr = 30 #TODO do not define arbitrary values, define a threshold based on the performance of the validation data metrics. 
        mse_thr = 0.4 #TODO Ibid.
        rmse_thr = 0.4 #TODO Ibid.

        # Classificatin section _______________________________________________________________________________________________________
        
        # Assign classes to each pixels in the prdiction
        # Assign the class 0 if the prediction is smaller than the threshold, else (value is bigger than threshold) assign 1 which mean there is an error
        binary_classified_map_mae = classify_pixels_with_confidence(test_maeMap, mae_thr)
        binary_classified_map_mse = classify_pixels_with_confidence(test_mseMap, mse_thr)
        binary_classified_map_rmse = classify_pixels_with_confidence(test_rmseMap, rmse_thr)

        # # Calculer les performances de la classification _________________________________
        # # Calculate TP = True possitif, FP = False possitif, TN = True Negatif, FN = False Negatif
        # tp, fp, tn, fn = binary_classification_analysis(binary_classified_map_mae, test_target_classification)
        # # Calculate precision, recall, accuracy and F1 score
        # precision_classification = precision(tp, fp)
        # recall_classification = recall(tp, fn)
        # accuracy_classification = accuracy(tp, fp, tn, fn)
        # f1_score_classification = f1score(precision_classification, recall_classification)
        # print(f'The classificaton performances are as follow: \n')
        # print(f'The precision is : {precision_classification}\nThe recall is : {recall_classification}\nThe accuracy is : {accuracy_classification}')
        # print(f'The f1 score is : {f1_score_classification}\n')

        # Calculer les performances de la classification _________________________________
        # Create ROC graph
        find_optimal_threshold_and_plot_roc(binary_classified_map_mae, test_target_classification, path_results)

        # Calculer les performances de la classification _________________________________
        # Calculate TP = True possitif, FP = False possitif, TN = True Negatif, FN = False Negatif
        tp, fp, tn, fn = binary_classification_analysis(binary_classified_map_mae, test_target_classification)
        # Calculate precision, recall, accuracy and F1 score
        precision_classification = precision(tp, fp)
        recall_classification = recall(tp, fn)
        accuracy_classification = accuracy(tp, fp, tn, fn)
        f1_score_classification = f1score(precision_classification, recall_classification)
        print(f'The classificaton performances are as follow: \n')
        print(f'The precision is : {precision_classification}\nThe recall is : {recall_classification}\nThe accuracy is : {accuracy_classification}')
        print(f'The f1 score is : {f1_score_classification}\n')





        # # Calculating metrics for the quality of the image regeneration (Global metrics, i.e for the whole image) for non defect images. 
        # mse_no_defects_global = []
        # rmse_no_defects_global = []
        # mae_no_defects_global = []
        # psnr_no_defects_global = []
        # ssim_no_defects_global = []

        # for (predict, target) in zip(result_no_defects_denorm, test_no_defects_denorm):
        #     # Calculating the metrics on the predict image. 
        #     mask = target != 0
        #     masked_predict = predict[mask]
        #     masked_target = target[mask]

        #     mse_no_defects_global.append(mean_squared_error(masked_target, masked_predict))
        #     rmse_no_defects_global.append(root_mean_squared_error(masked_target, masked_predict))
        #     mae_no_defects_global.append(mean_absolute_error(masked_target, masked_predict))

        #     psnr_no_defects_global.append(peak_signal_noise_ratio(target, predict))

        #     data_range = max(np.max(target), np.max(predict)) - min(np.min(target), np.min(predict))
        #     ssim_no_defects_global.append(structural_similarity(target, predict, data_range=data_range, channel_axis=2))
        
        # # Calculate the mean value for each metrics
        # mean_mse_no_defects_global = np.mean(np.array(mse_no_defects_global))
        # mean_rmse_no_defects_global = np.mean(np.array(rmse_no_defects_global))
        # mean_mae_no_defects_global = np.mean(np.array(mae_no_defects_global))
        # mean_psnr_no_defects_global = np.mean(np.array(psnr_no_defects_global))
        # mean_ssim_no_defects_global = np.mean(np.array(ssim_no_defects_global))

        # print(f'The average metrics for the whole dataset for the images without defect are: ')
        # print(f'mse: {mean_mse_no_defects_global}\nrmse: {mean_rmse_no_defects_global}\nmae: {mean_mae_no_defects_global}')
        # print(f'psnr: {mean_psnr_no_defects_global}\nssim: {mean_ssim_no_defects_global}\n\n')

        # # Calculating metrics for the quality of the image regeneration (Global metrics, i.e for the whole image) for defect images without considering the defect position. 
        # mse_defects_global = []
        # rmse_defects_global = []
        # mae_defects_global = []
        # psnr_defects_global = []
        # ssim_defects_global = []

        # for (predict, target) in zip(result_defects_denorm, test_defects_denorm):
        #     # Calculating the metrics on the predict image. 
        #     mask = target != 0
        #     masked_predict = predict[mask]
        #     masked_target = target[mask]

        #     mse_defects_global.append(mean_squared_error(masked_target, masked_predict))
        #     rmse_defects_global.append(root_mean_squared_error(masked_target, masked_predict))
        #     mae_defects_global.append(mean_absolute_error(masked_target, masked_predict))

        #     psnr_defects_global.append(peak_signal_noise_ratio(target, predict))

        #     min_predict = min(np.min(target), np.min(predict))
        #     max_predict = max(np.max(target), np.max(predict))
        #     data_range = max_predict - min_predict
        #     ssim_defects_global.append(structural_similarity(target, predict, data_range=data_range, channel_axis=2))
        
        # # Calculate the mean value for each metrics
        # mean_mse_defects_global = np.mean(np.array(mse_defects_global))
        # mean_rmse_defects_global = np.mean(np.array(rmse_defects_global))
        # mean_mae_defects_global = np.mean(np.array(mae_defects_global))
        # mean_psnr_defects_global = np.mean(np.array(psnr_defects_global))
        # mean_ssim_defects_global = np.mean(np.array(ssim_defects_global))

        # print(f'The average metrics for the whole dataset for the images with defect are: ')
        # print(f'mse: {mean_mse_defects_global}\nrmse: {mean_rmse_defects_global}\nmae: {mean_mae_defects_global}')
        # print(f'psnr: {mean_psnr_defects_global}\nssim: {mean_ssim_defects_global}\n\n')

        # # Creating boxplots for every metrics for the whole image, without taking in consideration where the defect are located:
        # metrics = [[mse_no_defects_global, mse_defects_global], [rmse_no_defects_global, rmse_defects_global], [mae_no_defects_global, mae_defects_global], [psnr_no_defects_global, psnr_defects_global], [ssim_no_defects_global, ssim_defects_global]]
        # name_metrics = ['mse', 'rmse', 'mae', 'psnr', 'ssim']
        # for (metric, name_metric) in zip(metrics, name_metrics):
        #     box_plot(metric, name_metric)
