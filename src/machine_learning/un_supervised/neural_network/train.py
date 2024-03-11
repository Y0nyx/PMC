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
import cv2

from keras import backend as K

import callbacks as cb
import data_processing as dp
import hyper_parameters_tuner as hp_tuner
import model as mod
import training_info as tr_info
from tensorflow.keras.models import save_model

debug = False
visualise = True

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

    parser.add_argument('--EPOCHS', type=int, default=100, 
                        help='Number of epoch used for the training of the model')
    parser.add_argument('--BATCH_SIZE', type=int, default=20,
                         help='Number of inputs that are processed in a single forward and backward pass during the training of the neural network')
    parser.add_argument('--LEARNING_RATE', type=float, default=0.001, 
                        help='Learning rate used when training the Neural Network with default value.')
    parser.add_argument('--PATH_RESULTS', type=str, default='../../../../Results/grosse_piece_seg_1', 
                        help='path where the results are going to be stored at.')
    parser.add_argument('--FILEPATH_WEIGHTS', type=str, default='/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/B_First_HP_Search/training_weights/',
                        help='Path where will be stored the weights.')
    parser.add_argument('--FILEPATH_WEIGHTS_SERCH', type=str, default='/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/B_First_HP_Search/training_weights/search_weights/',
                        help='Path where will be stored the weights for the hp search.')
    parser.add_argument('--HP_SEARCH', type=str, default='/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/aes_defect_detection/B_First_HP_Search',
                         help='Path to store the results of the hyper-parameter search.')
    parser.add_argument('--HP_NAME', type=str, default='hp_search_results',
                        help='Name used for karas tuner.')
    parser.add_argument('--MONITOR_LOSS', type=str, default='mean_absolute_error',
                         help='The metric being used for the loss function.')
    parser.add_argument('--MONITOR_METRIC', type=str, default ='mean_squared_error', 
                        help='The metric that is being monitored.')
    parser.add_argument('--MODE_METRIC', type=str, default='min',
                        help='The mode in which the value will be monitored.')
    parser.add_argument('--VERBOSE', action='store_true', default=True,
                        help='lag if the user want the code to print to help debug.')
    parser.add_argument('--DO_HP_SEARCH', action='store_true', default=True,
                        help='Flag used to see if the user is doing a hp search or not.')
    parser.add_argument('--EPOCHS_HP', type=int, default=100, 
                        help='Number of epoch used for the training of the model with the hp search.')
    parser.add_argument('--NUM_TRIALS_HP', type=int, default=50, 
                        help='Number of try to find the best hyper-parameters combinaison.')
    parser.add_argument('--EXECUTION_PER_TRIAL_HP', type=int, default= 2, 
                        help='Number of time the same hyper-parameters combinaison will be tested.')
    parser.add_argument('--NBEST', type=int, default=10, 
                        help='Number of best hp search that will be taken to train the model with.')
    parser.add_argument('--DATA_PATH', type=str, default='/home/jean-sebastien/Documents/s7/PMC/Data/images_cam_123/sub_images', 
                        help='Path where are located the data.')
    parser.add_argument('--MAX_PIXEL_VALUE', type=int, default=255,
                        help='Maximum pixel value for the analysed original image.')
    parser.add_argument('--SUB_WIDTH', type=int, default=256,
                        help='Width of the image after the subtitution of the images')
    parser.add_argument('--SUB_HEIGHT', type=int, default=256,
                        help='Width of the image after the subtitution of the images')

    return parser.parse_args()
    
class ModelTrainer:
    def __init__(self, train_input, train_input_loss, valid_input, valid_label, verbose, mode_metric, monitor_metric, monitor_loss, image_dimentions):
        self.input_train_norm = train_input
        self.input_train_label = train_input_loss
        self.input_valid_norm = valid_input
        self.input_valid_label = valid_label
        self.verbose = verbose
        self.mode_metric = mode_metric
        self.monitor_metric = monitor_metric
        self.monitor_loss = monitor_loss
        self.image_dimentions = image_dimentions

    def train_hp(self, epochs_hp, num_trials_hp, execution_per_trial_hp, path_results, nbest, hp_search):
        """
        Here we are doing an hp search and training the model with the N best results. 
        """
        callback_search = cb.TrainingCallbacks(filepath_weights_search, self.monitor_metric, self.mode_metric, self.verbose)
        callbacks_list_search = callback_search.get_callbacks(None)

        hp_tuner_instance = hp_tuner.KerasTuner(self.input_train_norm, self.input_train_label, self.input_valid_norm, self.input_valid_label, epochs_hp, num_trials_hp, 
                                                execution_per_trial_hp, self.mode_metric, self.verbose, callbacks_list_search, self.monitor_metric, self.monitor_loss, self.image_dimentions)
        hp_search_done = hp_tuner_instance.get_hp_search(hp_search, hp_name)

        #Store the results of the search
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        training_info = tr_info.TrainingInformation()
        training_info.write_hp_csv(path_results, hp_search_done, self.monitor_metric)

        #Train the N best HP
        best_hp = hp_search_done[:nbest]
        for j, trial in enumerate(best_hp, start=1):
            hp = trial.hyperparameters
            model = mod.AeModels(hp.get('lr'), self.monitor_loss, self.monitor_metric, self.image_dimentions)
            build_model = model.aes_defect_detection()

            name = f'model{j}'
            callback = cb.TrainingCallbacks(filepath_weights, self.monitor_metric, self.mode_metric, self.verbose)
            callbacks_list = callback.get_callbacks(name)

            history = train(build_model, self.input_train_norm, self.input_train_label, self.input_valid_norm, self.input_valid_label, int(1.2*epochs_hp), hp.get('batch_size'), callbacks_list)
            training_info.plot_graph(history, name, path_results, self.monitor_metric)

            # Nettoyage de la session Keras et collecte des dechets
            K.clear_session()
            gc.collect()
        
    def train_normal(self, epochs, batch_size, learning_rate, path_results):
        """
        Here we are training the model with the default parameters given in the initial variables. 
        """
        name = 'default_param'
        callback = cb.TrainingCallbacks(filepath_weights, self.monitor_metric, self.mode_metric, self.verbose)
        callbacks_list = callback.get_callbacks(name)
        
        model = mod.AeModels(learning_rate, self.monitor_loss, self.monitor_metric, self.image_dimentions)
        build_model = model.aes_defect_detection()
        history = train(build_model, self.input_train_norm, self.input_train_label, self.input_valid_norm, self.input_valid_label, epochs, batch_size, callbacks_list)

        training_info = tr_info.TrainingInformation()
        training_info.plot_graph(history, name, path_results, self.monitor_metric)

        # Save the model
        model_path = f"{path_results}/{name}_model.keras"
        save_model(build_model, model_path)
        print(f"Model saved to {model_path}")

        if debug:
            predictions = build_model.predict(self.input_valid_norm)
    
            predictions*=max_pixel_value
    
            output_dir = "./predictions"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
    
            for i in range(200):
                cv2.imwrite(f"{output_dir}/output_{i}.png", predictions[i])
    

if __name__ == '__main__':
    args = argparser()

    epochs = args.EPOCHS   
    batch_size = args.BATCH_SIZE
    learning_rate = args.LEARNING_RATE
    path_results = args.PATH_RESULTS
    filepath_weights = args.FILEPATH_WEIGHTS
    filepath_weights_search = args.FILEPATH_WEIGHTS_SERCH
    hp_search = args.HP_SEARCH
    hp_name = args.HP_NAME
    monitor_loss = args.MONITOR_LOSS
    monitor_metric = args.MONITOR_METRIC
    mode_metric = args.MODE_METRIC
    verbose = args.VERBOSE
    do_hp_search = args.DO_HP_SEARCH
    epochs_hp = args.EPOCHS_HP
    num_trials_hp = args.NUM_TRIALS_HP
    execution_per_trial_hp = args.EXECUTION_PER_TRIAL_HP
    nbest = args.NBEST
    data_path = args.DATA_PATH
    max_pixel_value = args.MAX_PIXEL_VALUE
    sub_width = args.SUB_WIDTH
    sub_height = args.SUB_HEIGHT

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if debug:
        #Quick tests
        do_hp_search = False
        data_path = "../../../../../Datasets/grosse_piece_seg_1"
        sub_width = 256
        sub_height = 256
        max_pixel_value = 255

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        except RuntimeError as e:
            print(f"Erreur lors de la configuration de la croissance de la mémoire du GPU: {e}")

    data_processing = dp.DataProcessing(sub_width, sub_height)
    train_input, train_input_loss, valid_input, test_input = data_processing.get_data_processing_stain(data_path, max_pixel_value) #TRAINING Change this line if you want to change the artificial defaut created. 

    if visualise:
        output_dir = "./output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i in range(10):
            cv2.imwrite(f"{output_dir}/train_input_{i}.png", (train_input[i]*max_pixel_value))
        for i in range(10):
            cv2.imwrite(f"{output_dir}/train_input_loss_{i}.png", train_input_loss[i]*max_pixel_value)
        for i in range(10):
            cv2.imwrite(f"{output_dir}/valid_input_{i}.png", valid_input[i]*max_pixel_value)
        # for i in range(10):
        #     cv2.imwrite(f"{output_dir}/valid_input_loss_{i}.png", valid_input_loss[i]*max_pixel_value)
        for i in range(10):
            cv2.imwrite(f"{output_dir}/test_input_{i}.png", test_input[i]*max_pixel_value)

    #DO NOT CHANGE THE CODE HERE AND FOR OTHER SECTIONS!
    _, row, column, channels = train_input.shape
    image_dimentions = (row, column, channels)

    train_model = ModelTrainer(train_input, train_input_loss, valid_input, valid_input, verbose, mode_metric, monitor_metric, monitor_loss, image_dimentions)
    if do_hp_search:
        history = train_model.train_hp(epochs_hp, num_trials_hp, execution_per_trial_hp, path_results, nbest, hp_search)
    else:
        history = train_model.train_normal(epochs, batch_size, learning_rate, path_results)

    print('The training is over and works as expected. You can now go test the Neural Network with train.sh script!')
