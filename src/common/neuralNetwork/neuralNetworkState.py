from enum import Enum


class NeuralNetworkState(Enum):
    """
    Enum for state machine of NeuralNetwork
    """
    INIT = 1
    READY = 2
    TRAIN = 3
    PREDICT = 4
    ERROR = 5
