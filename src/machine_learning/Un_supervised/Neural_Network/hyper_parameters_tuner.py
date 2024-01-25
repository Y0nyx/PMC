import keras_tuner
from keras_tuner import BayesianOptimization
from kerastuner.engine import multi_execution_tuner

import callbacks as cb
import model as mod

class KerasTuner():
    def __init__(self, input_train, input_test, epochs, num_trials, verbose):
        self.input_train = input_train
        self.input_test = input_test
        self.epochs = epochs
        self.num_trials = num_trials
        self.verbose = verbose

    def tuner_initializer(self, hypermodel: str, monitor: str, mode: str, num_trials: int, 
                          executions_per_trial: int, directory: str, directory_hp: str) -> BayesianOptimization:
        """
        Initialization of a Bayesian optimization tuner for hyperparameter tuning. 
        """
        return BayesianOptimization(
            hypermodel=hypermodel,
            objective=keras_tuner.Objective(monitor, direction=mode),
            max_trials=num_trials,
            executions_per_trial=executions_per_trial,
            overwrite=True,
            directory=directory,
            project_name=directory_hp
        )
    
    def hyper_parameter_search(self, hp):
        encoding_dim=hp.Choice('encoding_dim', values=[8, 16, 24, 32, 40])
        lr=hp.Choice('lr', values=[0.0001, 0.001])
        batch_size=hp.Choice('batch_size', values=[16, 32, 64, 128])

        trained_model = mod.AeModels(learning_rate=lr)
        model = trained_model.build_francois_chollet_autoencoder(encoding_dim=encoding_dim)

        return batch_size, model
    
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameter
        batch_size, model = self.hyper_parameter_search(hp)
        kwargs['batch_size'] = batch_size
        history = model.fit(*args, **kwargs)
        keys = list(history.history.keys())
        best_metric = min(keys[3])
        self.oracle.update_trial(trial.trial_id, {keys[3]: best_metric})

    def tuner_search(self, tuner):
        return tuner.search(
            x=self.input_train,
            y=self.input_train,
            epochs=self.epochs,
            validation_data=(self.input_test, self.input_test),
            callbacks=None,
            verbose=self.verbose,
            shuffle=True
        )
    
    def get_hp_search(self, hypermodel, monitor, mode, num_trials, executions_per_trial, directory, directory_hp):
        tuner = self.tuner_initializer(hypermodel, monitor, mode, num_trials, executions_per_trial, directory, directory_hp)
        multi_execution_tuner.MultiExecutionTuner.run_trial = self.run_trial
        callbacks = cb.TrainingCallbacks(filepath='/home/jean-sebastien/Documents/s7/PMC/results_ae/best_model_weights', monitor='mean_absolute_error', mode='val_min', verbose=1)
        _, early_stop, reduce_lr_plateau, end_nan = callbacks.get_callbacks()
        tuner = self.tuner_search(tuner)

        return tuner.oracle.get_best_trials(num_trials=self.num_trials)
         