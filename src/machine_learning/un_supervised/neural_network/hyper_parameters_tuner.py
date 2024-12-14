#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/25/2024
Description: Utilization of keras_tuner to find the best hyper-parameter combinaison. 
"""

import gc
import keras_tuner
import wandb

from keras import backend as K
from keras_tuner import BayesianOptimization, Objective
from wandb.keras import WandbCallback

import model as mod

class MyHyperModel(keras_tuner.HyperModel):  
    def __init__(self, train_input, train_label, valid_input, valid_label, epochs, callbacks, monitor_metric, monitor_loss, image_dimentions, strategy): 
            self.train_input = train_input
            self.train_label = train_label
            self.valid_input = valid_input
            self.valid_label = valid_label
            self.epochs = epochs
            self.callbacks = [callbacks[1:]]
            self.monitor_metric = monitor_metric
            self.monitor_loss = monitor_loss
            self.image_dimentions = image_dimentions
            self.strategy = strategy
 
    def build(self, hp):
        """
        Build the model with the HP to test. 
        """
        lr = hp.Choice('lr', values=[0.001])
        batch_size = hp.Int('batch_size', 32, 32, step=6, default=1)

        with self.strategy.scope():
            model_to_build = mod.AeModels(lr, self.monitor_loss, self.monitor_metric, self.image_dimentions)
            model = model_to_build.aes_defect_detection()

        return model
    
class CustomBayesianTuner(BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        batch_size = hp.get('batch_size')

        kwargs.pop('batch_size', None)
        kwargs.pop('validation_data', None)
        kwargs['batch_size'] = batch_size
        
        model = self.hypermodel.build(hp)

        wandb.init(project='aes_defect_detection_BTest', entity='dofa_unsupervised', config=trial.hyperparameters.values, mode="online", dir='/home/jean-sebastien/Documents/s7/PMC/results_un_supervised')

        callbacks = self.hypermodel.callbacks.copy() 
        kwargs.pop('callbacks', None)
        callbacks.append(WandbCallback(save_model=False))

        history = model.fit(        
            x=self.hypermodel.train_input,
            y=self.hypermodel.train_label,
            validation_data=(self.hypermodel.valid_input, self.hypermodel.valid_label),
            callbacks = callbacks,
            **kwargs
        )

        best_metric = min(history.history[self.hypermodel.monitor_metric]) 
        self.oracle.update_trial(trial.trial_id, {self.hypermodel.monitor_metric : best_metric}) 
        wandb.log({'best_metric': best_metric})

        wandb.finish()

        #Data cleaning after each try
        del model  
        K.clear_session()  
        gc.collect()  

    
class KerasTuner():
    def __init__(self, input_train_norm, input_train_norm_label, input_valid_norm, input_valid_norm_label, epochs, num_trials, executions_per_trial, mode, verbose, callbacks, monitor_metric, monitor_loss, image_dimentions, strategy):
        self.train_input = input_train_norm
        self.train_label = input_train_norm_label
        self.valid_input = input_valid_norm
        self.valid_label = input_valid_norm_label
        self.epochs = epochs
        self.num_trials = num_trials
        self.executions_per_trial = executions_per_trial
        self.mode = mode 
        self.verbose = verbose
        self.callbacks = callbacks
        self.monitor_metric = monitor_metric
        self.monitor_loss = monitor_loss
        self.image_dimentions = image_dimentions
        self.strategy = strategy

    def tuner_initializer(self, hp_search, hp_name) -> BayesianOptimization:
        """
        Initialization of a Bayesian optimization tuner for hyperparameter tuning. 
        """
        hypermodel = MyHyperModel(
            train_input = self.train_input,
            train_label = self.train_label,
            valid_input = self.valid_input,
            valid_label = self.valid_label,
            epochs = self.epochs, 
            callbacks = [self.callbacks[1:]],
            monitor_metric = self.monitor_metric,
            monitor_loss = self.monitor_loss,
            image_dimentions = self.image_dimentions,
            strategy = self.strategy
        )

        tuner = CustomBayesianTuner(  
            hypermodel=hypermodel,
            objective=Objective(self.monitor_metric, direction=self.mode),
            max_trials=self.num_trials,
            executions_per_trial=self.executions_per_trial,
            overwrite=True,
            directory=hp_search,
            project_name=hp_name
        )

        return tuner
    
    def tuner_search(self, tuner):
        """
        Manages the execution of multiple trials, each involving training a model once per trial with a specific set 
        of hyperparameters. 
        """
        tuner.search(
            self.train_input,
            self.train_label,
            epochs=self.epochs,
            validation_data=(self.valid_input, self.valid_label),
            verbose=self.verbose,
            shuffle=True
        )

        return tuner
    
    def get_hp_search(self, hp_search, hp_name):
        """
        Do the hp search and act as the main for hyper_parameters_tuners. 
        """
        tuner = self.tuner_initializer(hp_search, hp_name)

        self.tuner_search(tuner)

        return tuner.oracle.get_best_trials(num_trials=self.num_trials)
    