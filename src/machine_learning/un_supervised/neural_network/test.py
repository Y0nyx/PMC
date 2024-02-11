#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 02/03/2024
Description: Testing the Neural Network. 
"""

import argparse
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

import model as mod

def load_data(DATA_PATH):

    images = []

    #Data loading
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".png"):
            img = image.load_img(f'{DATA_PATH}/{filename}', target_size=(256, 256))
            images.append(image.img_to_array(img))
    images = np.array(images)

    print("=====================================")
    print("Loaded image np.array of shape: ", images.shape)
    print("=====================================")

    # Split the dataset into training and testing sets (70/30 split)
    input_train, input_test = train_test_split(images, train_size=0.8, test_size=0.2, random_state=59)
    print("=====================================")
    print("Splitted dataset in arrays of shape: ", input_train.shape, " | ", input_test.shape)
    print("=====================================")

    del images

    train_augmented = apply_random_blackout(input_train)
    test_augmented = apply_random_blackout(input_test)
    print("=====================================")
    print("Augmented splitted dataset in arrays of shape: ", train_augmented.shape, " | ", test_augmented.shape)
    print("=====================================")

    #Normalizing the data (0-1)
    input_train_norm, input_test_norm = normalize(input_train, input_test)
    input_train_aug_norm, input_test_aug_norm = normalize(train_augmented, test_augmented)

    return input_train_norm, input_train_aug_norm, input_test_norm, input_test_aug_norm

def apply_random_blackout(images, blackout_size=(32, 32)):
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

def normalize(input_train, input_test):
    input_train = input_train.astype('float32') / 255.
    input_test = input_test.astype('float32') / 255.
    # input_train = input_train.reshape((len(input_train), np.prod(input_train.shape[1:]))) #For MNIST only. 
    # input_test = input_test.reshape((len(input_test), np.prod(input_test.shape[1:])))

    return input_train, input_test

def de_normalize(input_test, result_test):
    difference_abs = np.abs(input_test - result_test)

    # Étape 1: Normalisation
    min_val = np.min(difference_abs)
    max_val = np.max(difference_abs)

    # Étape 2 et 3: Ajustement des valeurs pour couvrir la gamme de 0 à 255
    normalized_diff = (difference_abs - min_val) / (max_val - min_val) * 255

    # Étape 4: Conversion en uint8
    difference_reshaped = normalized_diff.astype('uint8')
    input_test = (input_test * 255).astype('uint8')
    result_test = (result_test * 255).astype('uint8')

    return input_test, result_test, difference_reshaped

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

def createPredImg(input_train, input_test, difference_reshaped, image, num, PATH_RESULTS):
    fig, axes = plt.subplots(1, 3)
    fig.suptitle('Input data vs Results')

    axes[0].set_title('Inputs')
    axes[0].imshow(input_train, cmap='gray')
    axes[1].set_title('Results')
    axes[1].imshow(input_test, cmap='gray')
    axes[2].set_title('difference')
    axes[2].imshow(difference_reshaped, cmap='jet')

    dir = f'{PATH_RESULTS}/image/result_model_{num+1}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}/{image+1}.png')
    plt.close()

def argparser():
    parser = argparse.ArgumentParser(description='Argument used in the code passed by the bash file.')

    parser.add_argument('--PATH_RESULTS', type=str, help='path where the results are going to be stored at.')
    parser.add_argument('--NBEST', type=int, help='Number of best hp search that will be taken to train the model with.')
    parser.add_argument('--NUM_TRAIN_REGENERATE', type=int, help='Number of test data that will be used for the generation.')
    parser.add_argument('--FILEPATH_WEIGHTS', type=str, help='Path where are stored the weights.')
    parser.add_argument('--DATA_PATH', type=str, help='Path where are located the data.')

    return parser.parse_args()

if __name__ =='__main__':
    args = argparser()

    input_train_norm, input_train_aug_norm, input_test_norm, input_test_aug_norm = load_data(args.DATA_PATH)

    data_frame = pd.read_csv(f'{args.PATH_RESULTS}/hp_search_results.csv')
    #TODO Might by a way to automatize this. 
    learning_rate = data_frame['lr']

    for j in range(args.NBEST):
        model = mod.AeModels(learning_rate=float(learning_rate[j]))
        build_model = model.build_basic_cae()   #Change this line if the model change. 

        name = f"model{j+1}"
        build_model.load_weights(f'{args.FILEPATH_WEIGHTS}/search_{name}')

        input_test_norm = input_test_norm[0:args.NUM_TRAIN_REGENERATE]

        result_norm = prediction(build_model, input_test_norm)

        input_test_denorm, result_denorm, difference_reshaped = de_normalize(input_test_norm, result_norm)

        for i in range(args.NUM_TRAIN_REGENERATE):
            createPredImg(input_test_denorm[i], result_denorm[i], difference_reshaped[i], i, j, args.PATH_RESULTS)
