#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/25/2024
Description: Callbacks used in the training of the model to minimize bad hyper-parameters(hp) combinaisons to
             maximize GPU utilization. A bad hp based on the monitor metrics which is in our case the mae.
"""

import tensorflow as tf

class TrainingCallbacks():
    def __init__(self, filepath: str, monitor: str, mode: str, verbose: int):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

    def best_model_weights(self, name) -> tf.keras.callbacks.Callback:
        """
        Modelcheckpoint callback to save the best model weights.
        """
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{self.filepath}/search_{name}', 
            monitor=self.monitor, 
            save_best_only=True,
            mode=self.mode,
            save_weights_only=True, 
            save_freq='epoch',
            initial_value_threshold=None
        )

    def early_stop(self, min_delta: float=0.001, patience: int=15) -> tf.keras.callbacks.Callback:
        """
        EarlyStopping callback.
        """
        return tf.keras.callbacks.EarlyStopping(
            monitor=self.monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=self.verbose,
            mode=self.mode
        )

    def reduce_lr_plateau(self, factor: float=0.1, patience: int=10, min_delta: float=0.001) -> tf.keras.callbacks.Callback:
        """
        Reduce lr on plateau callback to reduce the learning rate when a plateau is detected.
        """
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.monitor, 
            factor=factor,
            patience=patience,
            verbose=self.verbose,
            mode=self.mode, 
            min_delta=min_delta,
            cooldown=0,
            min_le=0
        )

    def end_nan(self) -> tf.keras.callbacks.Callback:
        """
        TerminateOnNaN cakkback to terminate training when NaN loss is encountered. 
        """
        return tf.keras.callbacks.TerminateOnNaN()

    def get_callbacks(self, name, min_delta_es: float=0.001, patience_es: int=15, factor_rp: float=0.1, 
                      patience_rp: int=10, min_delta_rp: float=0.001) -> list:
        return [
            self.best_model_weights(name),
            self.early_stop(min_delta_es, patience_es),
            self.reduce_lr_plateau(factor_rp, patience_rp, min_delta_rp),
            self.end_nan()
        ]
    