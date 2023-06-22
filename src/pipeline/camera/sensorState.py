from enum import Enum


class SensorState(Enum):
    """
    Enum for state machine of Sensor
    """
    INIT = 1
    READY = 2
    ERROR = 3
