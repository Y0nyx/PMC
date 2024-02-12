#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/25/2024
Description: Training Neural Network.
"""

import argparse
import csv
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras import backend as K
import gc
import math

import callbacks as cb
import hyper_parameters_tuner as hp_tuner
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

    # Split the dataset into training and testing sets (90/10 split)
    input_train_valid, input_test = train_test_split(images, train_size=0.9, test_size=0.1, random_state=59)
    #Create a validation set
    len_separation = int( math.floor(len(input_train_valid) * 0.91) )
    input_train = input_train_valid[:len_separation]
    input_valid = input_train_valid[len_separation:]
    print("=====================================")
    print(f'The sata is splitted into three set. TRAIN, VALIDATION and TEST')
    print("Splitted dataset in arrays of shape: ", input_train.shape, " | ", input_valid.shape, " | ", input_test.shape)
    print("=====================================")

    del images

    train_augmented = apply_random_blackout(input_train)
    valid_augmented = apply_random_blackout(input_valid)
    test_augmented = apply_random_blackout(input_test)
    print("=====================================")
    print("Augmented splitted dataset in arrays of shape: ", train_augmented.shape, " | ",valid_augmented.shape, " | ", test_augmented.shape)
    print("=====================================")

    #Normalizing the data (0-1)
    input_train_norm, input_valid_norm, input_test_norm = normalize(input_train, input_valid, input_test)
    input_train_aug_norm, input_valid_aug_norm, input_test_aug_norm = normalize(train_augmented, valid_augmented, test_augmented)

    return input_train_norm, input_valid_norm, input_test_norm, input_train_aug_norm, input_valid_aug_norm, input_test_aug_norm

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

def normalize(input_train, input_valid, input_test):
    input_train = input_train.astype('float32') / 255.
    input_valid = input_valid.astype('float32') / 255.
    input_test = input_test.astype('float32') / 255.

    return input_train, input_valid, input_test

def train(model, input_train, input_train_aug, input_test, input_test_aug, epochs, batch_size, callbacks):
    history = model.fit(
        x=input_train_aug, 
        y=input_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        validation_data=(input_test_aug, input_test),
        shuffle=True
    )

    return history

def write_hp_csv(dir, n_best_hp):
    fieldnames = ['model_rank', 'lr', 'batch_size', 'metric_loss']
    with open(f'{dir}/hp_search_results.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, trial in enumerate(n_best_hp):
            hps = trial.hyperparameters
            validation_loss = trial.metrics.get_best_value('mean_absolute_error')
            row_dict = {
                'model_rank': i+1, 
                'lr': hps.get('lr'),
                'batch_size': hps.get('bFirstatch_size'),
                'metric_loss': validation_loss
            }
            writer.writerow(row_dict)

def plot_graph(history, name, directory):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'blue', label='Training_loss')
    plt.plot(epochs, val_loss, 'orange', label='validation_loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    

    dir = directory + '/training'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}/train_val_loss_{name}.png')
    plt.close()

    acc = history.history['mean_absolute_error']
    val_acc = history.history['val_mean_absolute_error']
    plt.plot(epochs, acc, 'blue', label='Training accuracy')
    plt.plot(epochs, val_acc, 'orange', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{dir}/train_val_acc_{name}.png')
    plt.close()

def argparser():
    parser = argparse.ArgumentParser(description='Argument used in the code passed by the bash file.')

    parser.add_argument('--EPOCHS', type=int, help='Number of epoch used for the training of the model')
    parser.add_argument('--BATCH_SIZE', type=int, help='Number of inputs that are processed in a single forward and backward pass during the training of the neural network')
    parser.add_argument('--PATH_RESULTS', type=str, help='path where the results are going to be stored at.')
    parser.add_argument('--FILEPATH_WEIGHTS', type=str, help='Path where will be stored the weights.')
    parser.add_argument('--FILEPATH_WEIGHTS_SERCH', type=str, help='Path where will be stored the weights for the hp search.')
    parser.add_argument('--HP_SEARCH', type=str, help='Path to store the results of the hyper-parameter search.')
    parser.add_argument('--HP_NAME', type=str, help='Name used for karas tuner.')
    parser.add_argument('--MONITOR_METRIC', type=str, help='The metric that is being monitored.')
    parser.add_argument('--MODE_METRIC', type=str, help='The mode in which the value will be monitored.')
    parser.add_argument('--VERBOSE', action='store_true', help='lag if the user want the code to print to help debug.')
    parser.add_argument('--DO_HP_SEARCH', action='store_true', help='Flag used to see if the user is doing a hp search or not.')
    parser.add_argument('--EPOCHS_HP', type=int, help='Number of epoch used for the training of the model with the hp search.')
    parser.add_argument('--NUM_TRIALS_HP', type=int, help='Number of try to find the best hyper-parameters combinaison.')
    parser.add_argument('--EXECUTION_PER_TRIAL_HP', type=int, help='Number of time the same hyper-parameters combinaison will be tested.')
    parser.add_argument('--NBEST', type=int, help='Number of best hp search that will be taken to train the model with.')
    parser.add_argument('--DATA_PATH', type=str, help='Path where are located the data.')

    return parser.parse_args()
    
class ModelTrainer:
    def __init__(self, input_train_norm, input_train_aug_norm, input_valid_norm, input_valid_aug_norm, VERBOSE, MODE_METRIC, MONITOR_METRIC):
        self.input_train_norm = input_train_norm
        self.input_train_aug_norm = input_train_aug_norm
        self.input_valid_norm = input_valid_norm
        self.input_valid_aug_norm = input_valid_aug_norm
        self.verbose = VERBOSE
        self.mode_metric = MODE_METRIC
        self.monitor_metric = MONITOR_METRIC

    def train_hp(self, EPOCHS_HP, NUM_TRIALS_HP, EXECUTION_PER_TRIAL_HP, PATH_RESULTS):
        """
        Here we are doing an hp search and training the model with the N best results. 
        """
        callback_search = cb.TrainingCallbacks(args.FILEPATH_WEIGHTS_SERCH, args.MONITOR_METRIC, args.MODE_METRIC, args.VERBOSE)
        callbacks_list_search = callback_search.get_callbacks(None)

        hp_tuner_instance = hp_tuner.KerasTuner(self.input_train_norm, self.input_train_aug_norm, self.input_valid_norm, self.input_valid_aug_norm, EPOCHS_HP, NUM_TRIALS_HP, 
                                                EXECUTION_PER_TRIAL_HP, self.monitor_metric, self.mode_metric, self.verbose, callbacks_list_search)
        hp_search = hp_tuner_instance.get_hp_search(args.HP_SEARCH, args.HP_NAME)

        #Store the results of the search
        if not os.path.exists(PATH_RESULTS):
            os.makedirs(PATH_RESULTS)
        write_hp_csv(PATH_RESULTS, hp_search)

        #Train the N best HP
        best_hp = hp_search[:args.NBEST]
        for j, trial in enumerate(best_hp, start=1):
            hp = trial.hyperparameters
            model = mod.AeModels(learning_rate=hp.get('lr'))
            build_model = model.build_basic_cae()

            name = f'model{j}'
            callback = cb.TrainingCallbacks(args.FILEPATH_WEIGHTS, args.MONITOR_METRIC, args.MODE_METRIC, args.VERBOSE)
            callbacks_list = callback.get_callbacks(name)

            history = train(build_model, self.input_train_norm, self.input_train_aug_norm, self.input_valid_norm, self.input_valid_aug_norm, int(1.2*EPOCHS_HP), 32, callbacks_list)
            plot_graph(history, name, PATH_RESULTS)

            # Nettoyage de la session Keras et collecte des déchets
            K.clear_session()
            gc.collect()
        
    def train(self, EPOCHS, BATCH_SIZE, PATH_RESULTS):
        """
        Here we are training the model with the default parameters given in the initial variables. 
        """
        model = mod.AeModels(learning_rate=0.001)
        build_model = model.build_basic_cae()
        history = train(build_model, self.input_train_norm, self.input_train_aug_norm, self.input_valid_norm, self.input_valid_aug_norm, EPOCHS, BATCH_SIZE, self.callbacks_list)

        name = 'default_param'
        plot_graph(history, name, PATH_RESULTS)
    

if __name__ == '__main__':
    args = argparser()

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        except RuntimeError as e:
            print(f"Erreur lors de la configuration de la croissance de la mémoire du GPU: {e}")

    input_train_norm, input_valid_norm, input_test_norm, input_train_aug_norm, input_valid_aug_norm, input_test_aug_norm = load_data(args.DATA_PATH)

    train_model = ModelTrainer(input_train_norm, input_train_aug_norm, input_valid_norm, input_valid_aug_norm, args.VERBOSE, args.MODE_METRIC, args.MONITOR_METRIC)
    if args.DO_HP_SEARCH:
        history = train_model.train_hp(args.EPOCHS_HP, args.NUM_TRIALS_HP, args.EXECUTION_PER_TRIAL_HP, args.PATH_RESULTS)
    else:
        history = train_model.train(args.EPOCHS, args.BATCH_SIZE, args.PATH_RESULTS)
