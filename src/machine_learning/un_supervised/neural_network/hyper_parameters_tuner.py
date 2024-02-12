#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/25/2024
Description: Utilization of keras_tuner to find the best hyper-parameter combinaison. 
"""

import gc
import keras_tuner

from keras import backend as K
from keras_tuner import BayesianOptimization, Objective

import model as mod

class MyHyperModel(keras_tuner.HyperModel):  
    def __init__(self, input_train, input_train_aug, input_test_aug, input_test, epochs, callbacks): 
        self.input_train_aug = input_train_aug, 
        self.input_train = input_train,
        self.input_test_aug = input_test_aug,
        self.input_test = input_test,
        self.epochs = epochs
        self.callbacks = callbacks
 
    def build(self, hp):
        """
        Build the model with the HP to test. 
        """
        encoding_dim = hp.Choice('encoding_dim', values=[8, 16, 24, 32, 40])
        lr = hp.Choice('lr', values=[0.0001, 0.001, 0.01])
        batch_size = hp.Int('batch_size', 2, 40, step=2, default=1)

        model_to_build = mod.AeModels(learning_rate=lr)
        model = model_to_build.build_basic_cae()

        return model
    
class CustomBayesianTuner(BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        batch_size = hp.get('batch_size')

        kwargs.pop('batch_size', None)
        kwargs.pop('validation_data', None)

        kwargs['batch_size'] = batch_size
        
        model = self.hypermodel.build(hp)

        history = model.fit(        
            x=self.hypermodel.input_train_aug,
            y=self.hypermodel.input_train,
            validation_data=(self.hypermodel.input_test_aug, self.hypermodel.input_test),
            **kwargs
        )

        best_metric = min(history.history['mean_absolute_error']) 
        self.oracle.update_trial(trial.trial_id, {'mean_absolute_error': best_metric}) 

        # Nettoyage après chaque essai
        del model  # Supprimer explicitement le modèle
        K.clear_session()  # Nettoyer la session Keras
        gc.collect()  # Forcer la collecte des déchets de Python

    
class KerasTuner():
    def __init__(self, input_train_norm, input_train_aug_norm, input_test_norm, input_test_aug_norm, epochs, num_trials, executions_per_trial, monitor, mode, verbose, callbacks):
        self.input_train = input_train_norm
        self.input_train_aug = input_train_aug_norm
        self.input_test = input_test_norm
        self.input_test_aug = input_test_aug_norm
        self.epochs = epochs
        self.num_trials = num_trials
        self.executions_per_trial = executions_per_trial
        self.monitor = monitor
        self.mode = mode 
        self.verbose = verbose
        self.callbacks = callbacks

    def tuner_initializer(self, HP_SEARCH, HP_NAME) -> BayesianOptimization:
        """
        Initialization of a Bayesian optimization tuner for hyperparameter tuning. 
        """
        hypermodel = MyHyperModel(
            input_train_aug = self.input_train_aug,
            input_train = self.input_train,
            input_test_aug = self.input_test_aug,
            input_test = self.input_test,
            epochs = self.epochs, 
            callbacks = self.callbacks[0]
        )

        tuner = CustomBayesianTuner(  
            hypermodel=hypermodel,
            objective=Objective(self.monitor, direction=self.mode),
            max_trials=self.num_trials,
            executions_per_trial=self.executions_per_trial,
            overwrite=True,
            directory=HP_SEARCH,
            project_name=HP_NAME
        )

        return tuner
    
    def tuner_search(self, tuner, callbacks):
        """
        Manages the execution of multiple trials, each involving training a model once per trial with a specific set 
        of hyperparameters. 
        """
        print("Splitted dataset in arrays of shape: ", self.input_train.shape, " | ", self.input_test.shape)
        print("Augmented splitted dataset in arrays of shape: ", self.input_train_aug.shape, " | ", self.input_test_aug.shape)
        tuner.search(
            self.input_train_aug,
            self.input_train,
            epochs=self.epochs,
            validation_data=(self.input_test_aug, self.input_test),
            callbacks=callbacks[1:],
            verbose=self.verbose,
            shuffle=True
        )

        return tuner
    
    def get_hp_search(self, HP_SEARCH, HP_NAME):
        """
        Do the hp search and act as the main for hyper_parameters_tuners. 
        """
        tuner = self.tuner_initializer(HP_SEARCH, HP_NAME)

        self.tuner_search(tuner, self.callbacks[1:])

        return tuner.oracle.get_best_trials(num_trials=self.num_trials)
    