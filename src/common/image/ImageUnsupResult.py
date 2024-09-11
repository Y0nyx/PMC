import numpy as np
from . import Image

class ImageUnsupResult(Image):
    """
    This class is built to store the resulting output of the unsupervised model. It contains all necessary information
    to interpret and validate the model's prediction.
    """

    def __init__(self, img: np.ndarray, pred: np.ndarray, ba_mask: np.ndarray):
        super().__init__(img)
        self._prediction = pred
        self._ba_mask = ba_mask

    @property
    def prediction(self) -> np.ndarray:
        """
        Get the value of the prediction.
        :return: Prediction array.
        """
        return self._prediction

    @prediction.setter
    def prediction(self, pred: np.ndarray) -> None:
        """
        Set the value of the prediction.
        :param pred: Prediction array.
        :return: None
        """
        self._prediction = pred

    @property
    def ba_mask(self) -> np.ndarray:
        """
        Get the value of the blackout mask.
        :return: Blackout mask array.
        """
        return self._ba_mask

    @ba_mask.setter
    def ba_mask(self, ba_mask: np.ndarray) -> None:
        """
        Set the value of the blackout mask.
        :param ba_mask: Blackout mask array.
        :return: None
        """
        self._ba_mask = ba_mask