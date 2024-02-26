#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 02/03/2024
Description: Testing the Neural Network. 
"""

import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd

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

def createPredImg(input_train, input_test, difference_reshaped, image, num, PATH_RESULTS):
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

    data_processing = dp.DataProcessing()
    train_input, train_input_loss, valid_input, test_input = data_processing.get_data_processing_stain(args.DATA_PATH)

    data_frame = pd.read_csv(f'{args.PATH_RESULTS}/hp_search_results.csv')
    #List all the hp used during training. 
    learning_rate = data_frame['lr']

    for j in range(args.NBEST):
        model = mod.AeModels(learning_rate=float(learning_rate[j]))
        build_model = model.aes_defect_detection()   #Change this line if the model change. 

        name = f"model{j+1}"
        build_model.load_weights(f'{args.FILEPATH_WEIGHTS}/search_{name}', by_name=True, skip_mismatch=True)

        input_test_norm = test_input[0:args.NUM_TRAIN_REGENERATE]

        result_norm = prediction(build_model, input_test_norm)

        input_test_denorm, result_denorm, difference_reshaped = data_processing.de_normalize(input_test_norm, result_norm)

        for i in range(args.NUM_TRAIN_REGENERATE):
            createPredImg(input_test_denorm[i], result_denorm[i], difference_reshaped[i], i, j, args.PATH_RESULTS)

    print('The testing of the Neural Network has been done corectly!')
