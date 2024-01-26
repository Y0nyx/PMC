import keras_tuner
from keras_tuner import BayesianOptimization, HyperParameters, Objective
from keras_tuner.engine.trial import Trial
from kerastuner.engine import multi_execution_tuner
import tensorflow as tf

import callbacks as cb
import model as mod

class KerasTuner():
    def __init__(self, input_train, input_test, epochs: int, num_trials: int, executions_per_trial: int, monitor: str, mode: str, verbose: int):
        self.input_train = input_train
        self.input_test = input_test
        self.epochs = epochs
        self.num_trials = num_trials
        self.executions_per_trial = executions_per_trial
        self.verbose = verbose
        self.monitor = monitor
        self.mode = mode 

    def tuner_initializer(self, hypermodel: str, directory: str, directory_hp: str) -> BayesianOptimization:
        """
        Initialization of a Bayesian optimization tuner for hyperparameter tuning. 
        """
        return BayesianOptimization(
            hypermodel=hypermodel,
            objective=Objective(self.monitor, direction=self.mode),
            max_trials=self.num_trials,
            executions_per_trial=self.executions_per_trial,
            overwrite=True,
            directory=directory,
            project_name=directory_hp
        )
    
    def hyper_parameter_search(self, hp: HyperParameters) -> (int, keras_tuner.HyperParameters.Model):
        """
        Tested hyper-parameters and model initialization. 
        """
        encoding_dim=hp.Choice('encoding_dim', values=[8, 16, 24, 32, 40])
        lr=hp.Choice('lr', values=[0.0001, 0.001])
        batch_size=hp.Choice('batch_size', values=[16, 32, 64, 128])

        trained_model = mod.AeModels(learning_rate=lr)
        model = trained_model.build_francois_chollet_autoencoder(encoding_dim=encoding_dim)

        return batch_size, model
    
    def run_trial(self, trial: Trial, *args, **kwargs) -> None:
        """
        Run the trial with the current set of hp and save batch_size information. 
        """
        hp = trial.hyperparameter
        batch_size, model = self.hyper_parameter_search(hp)
        kwargs['batch_size'] = batch_size
        history = model.fit(*args, **kwargs)
        keys = list(history.history.keys())
        best_metric = min(keys[3])
        self.oracle.update_trial(trial.trial_id, {keys[3]: best_metric})

    def tuner_search(self, tuner: Trial, callbacks: list[tf.keras.callbacks.Callback]) -> None:
        """
        Manages the execution of multiple trials, each involving training a model once per trial with a specific set 
        of hyperparameters. 
        """
        return tuner.search(
            x=self.input_train,
            y=self.input_train,
            epochs=self.epochs,
            validation_data=(self.input_test, self.input_test),
            callbacks=callbacks,
            verbose=self.verbose,
            shuffle=True
        )
    
    def get_hp_search(self, directory: str, directory_hp: str):
        """
        Do the hp search and act as the main for hyper_parameters_tuners. 
        """
        tuner = self.tuner_initializer(self.hyper_parameter_search, self.monitor, self.mode, self.num_trials, self.executions_per_trial, directory, directory_hp)
        multi_execution_tuner.MultiExecutionTuner.run_trial = self.run_trial()

        callbacks_instance = cb.TrainingCallbacks(None, self.monitor, self.mode, self.verbose)
        callbacks_list = callbacks_instance.get_callbacks()
        callbacks_list = callbacks_list[1:]

        tuner = self.tuner_search(tuner, callbacks_list)

        return tuner.oracle.get_best_trials(num_trials=self.num_trials)
         