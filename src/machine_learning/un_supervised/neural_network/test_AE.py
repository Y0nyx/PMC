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

import model as mod

def normalize(input_train, input_test):
    input_train = input_train.astype('float32') / 255.
    input_test = input_test.astype('float32') / 255.
    input_train = input_train.reshape((len(input_train), np.prod(input_train.shape[1:])))
    input_test = input_test.reshape((len(input_test), np.prod(input_test.shape[1:])))

    return input_train, input_test

def de_normalize(input_test, result_test):
    original_image_height = 28
    original_image_width = 28

    input_test = input_test.reshape((len(input_test), original_image_height, original_image_width))
    result_test = result_test.reshape((len(result_test), original_image_height, original_image_width))
    input_test = input_test * 255
    result_test = result_test * 255

    return input_test, result_test

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

def createPredImg(input_train, input_test, image, num, PATH_RESULTS):
    # Redimensionner les images de 784 à 28x28
    input_train_reshaped = np.reshape(input_train, (28, 28))
    input_test_reshaped = np.reshape(input_test, (28, 28))
    difference_reshaped = input_train_reshaped - input_test_reshaped
    fig, axes = plt.subplots(1, 3)
    fig.suptitle('Input data vs Results')

    axes[0].set_title('Inputs')
    axes[0].imshow(input_train_reshaped, cmap='gray')
    axes[1].set_title('Results')
    axes[1].imshow(input_test_reshaped, cmap='gray')
    axes[2].set_title('difference')
    axes[2].imshow(difference_reshaped, cmap='gray')

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

    return parser.parse_args()

if __name__ =='__main__':
    args = argparser()

    (input_train, _), (input_test, _) = mnist.load_data()
    input_train_norm, input_test_norm = normalize(input_train, input_test)

    data_frame = pd.read_csv(f'{args.PATH_RESULTS}/hp_search_results.csv')
    #TODO Might by a way to automatize this. 
    learning_rate = data_frame['lr']

    for j in range(args.NBEST):
        model = mod.AeModels(learning_rate=learning_rate[j])
        build_model = model.build_francois_chollet_autoencoder(input_shape=(784,), encoding_dim=32)    #Change this line if the model change. 

        name = f"model{j+1}"
        build_model.load_weights(f'{args.FILEPATH_WEIGHTS}/search_{name}')

        input_test_norm = input_test_norm[0:args.NUM_TRAIN_REGENERATE]

        result_norm = prediction(build_model, input_test_norm)

        input_test_denorm, result_denorm = de_normalize(input_test_norm, result_norm)

        for i in range(args.NUM_TRAIN_REGENERATE):
            createPredImg(input_test_denorm[i], result_denorm[i], i, j, args.PATH_RESULTS)
