#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/25/2024
Description: Training Neural Network.
"""

import argparse
import gc
import os
import tensorflow as tf

from keras import backend as K

import callbacks as cb
import data_processing as dp
import hyper_parameters_tuner as hp_tuner
import model as mod
import training_info as tr_info

def train(model, input_train, input_label, valid_input, valid_label, epochs, batch_size, callbacks):
    history = model.fit(
        x=input_train, 
        y=input_label,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        validation_data=(valid_input, valid_label),
        shuffle=True
    )

    return history

def argparser():
    parser = argparse.ArgumentParser(description='Argument used in the code passed by the bash file.')

    parser.add_argument('--EPOCHS', type=int, help='Number of epoch used for the training of the model')
    parser.add_argument('--BATCH_SIZE', type=int, help='Number of inputs that are processed in a single forward and backward pass during the training of the neural network')
    parser.add_argument('--PATH_RESULTS', type=str, help='path where the results are going to be stored at.')
    parser.add_argument('--FILEPATH_WEIGHTS', type=str, help='Path where will be stored the weights.')
    parser.add_argument('--FILEPATH_WEIGHTS_SERCH', type=str, help='Path where will be stored the weights for the hp search.')
    parser.add_argument('--HP_SEARCH', type=str, help='Path to store the results of the hyper-parameter search.')
    parser.add_argument('--HP_NAME', type=str, help='Name used for karas tuner.')
    parser.add_argument('--MONITOR_LOSS', type=str, help='The metric being used for the loss function.')
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
    def __init__(self, train_input, train_input_loss, valid_input, valid_label, VERBOSE, MODE_METRIC, MONITOR_METRIC):
        self.input_train_norm = train_input
        self.input_train_label = train_input_loss
        self.input_valid_norm = valid_input
        self.input_valid_label = valid_label
        self.verbose = VERBOSE
        self.mode_metric = MODE_METRIC
        self.monitor_metric = MONITOR_METRIC

    def train_hp(self, EPOCHS_HP, NUM_TRIALS_HP, EXECUTION_PER_TRIAL_HP, PATH_RESULTS):
        """
        Here we are doing an hp search and training the model with the N best results. 
        """
        callback_search = cb.TrainingCallbacks(args.FILEPATH_WEIGHTS_SERCH, args.MONITOR_METRIC, args.MODE_METRIC, args.VERBOSE)
        callbacks_list_search = callback_search.get_callbacks(None)

        hp_tuner_instance = hp_tuner.KerasTuner(self.input_train_norm, self.input_train_label, self.input_valid_norm, self.input_valid_label, EPOCHS_HP, NUM_TRIALS_HP, 
                                                EXECUTION_PER_TRIAL_HP, self.monitor_metric, self.mode_metric, self.verbose, callbacks_list_search, args.MONITOR_METRIC)
        hp_search = hp_tuner_instance.get_hp_search(args.HP_SEARCH, args.HP_NAME)

        #Store the results of the search
        if not os.path.exists(PATH_RESULTS):
            os.makedirs(PATH_RESULTS)
        training_info = tr_info.TrainingInformation()
        training_info.write_hp_csv(PATH_RESULTS, hp_search, args.MONITOR_METRIC)

        #Train the N best HP
        best_hp = hp_search[:args.NBEST]
        for j, trial in enumerate(best_hp, start=1):
            hp = trial.hyperparameters
            model = mod.AeModels(learning_rate=hp.get('lr'))
            build_model = model.aes_defect_detection()

            name = f'model{j}'
            callback = cb.TrainingCallbacks(args.FILEPATH_WEIGHTS, args.MONITOR_METRIC, args.MODE_METRIC, args.VERBOSE)
            callbacks_list = callback.get_callbacks(name)

            history = train(build_model, self.input_train_norm, self.input_train_label, self.input_valid_norm, self.input_valid_label, int(1.2*EPOCHS_HP), hp.get('batch_size'), callbacks_list)
            training_info.plot_graph(history, name, PATH_RESULTS, args.MONITOR_METRIC)

            # Nettoyage de la session Keras et collecte des dechets
            K.clear_session()
            gc.collect()
        
    def train_normal(self, EPOCHS, BATCH_SIZE, PATH_RESULTS):
        """
        Here we are training the model with the default parameters given in the initial variables. 
        """
        model = mod.AeModels(learning_rate=0.001)
        build_model = model.aes_defect_detection()
        history = train(build_model, self.input_train_norm, self.input_train_label, self.input_valid_norm, self.input_valid_aug_norm, EPOCHS, BATCH_SIZE, self.callbacks_list)

        training_info = tr_info.TrainingInformation()
        name = 'default_param'
        training_info.plot_graph(history, name, PATH_RESULTS, args.MONITOR_METRIC)
    

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

    data_processing = dp.DataProcessing()
    train_input, train_input_loss, valid_input, test_input = data_processing.get_data_processing_stain(args.DATA_PATH)

    #train_model = ModelTrainer(train_input, train_input_loss, valid_input, valid_input, args.VERBOSE, args.MODE_METRIC, args.MONITOR_METRIC)
    #if args.DO_HP_SEARCH:
        #history = train_model.train_hp(args.EPOCHS_HP, args.NUM_TRIALS_HP, args.EXECUTION_PER_TRIAL_HP, args.PATH_RESULTS)
    #else:
        #history = train_model.train_normal(args.EPOCHS, args.BATCH_SIZE, args.PATH_RESULTS)

    #print('The training is over and works as expected. You can now go test the Neural Network with train.sh script!')
