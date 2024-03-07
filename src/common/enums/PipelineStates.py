from enum import Enum


class PipelineState(Enum):
    INIT = 0
    TRAINING = 1
    DATASET = 2
    ANALYSING = 3
