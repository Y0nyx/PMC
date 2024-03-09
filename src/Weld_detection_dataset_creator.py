from pipeline.models.Model import YoloModel
from pipeline.data.DataManager import DataManager
from common.enums.PipelineStates import PipelineState
from common.image.Image import Image

import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image as Img
import numpy as np

from ultralytics import YOLO
#from clearml import Task
from common.image.ImageCollection import ImageCollection
from common.utils import DataManager as Mock_DataManager
from common.Constants import *
from TrainingManager import TrainingManager

import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
dataset_path = "../../Datasets/grosse_piece"

class Pipeline:

    def __init__(self, models, verbose: bool = True, State: PipelineState= PipelineState.INIT):
        self.verbose = verbose

        self.print("=== Init Pipeline ===")  # Fixed this line

        self.models = []
        for model in models:
            self.models.append(model)

        self._state = State

        self._trainingManager = TrainingManager(is_time_threshold=False, verbose=self.verbose)

    def detect(self, show: bool = False, save: bool = False, conf: float = 0.65):
        images = []

        print("loading images")
        # Data loading
        for filename in os.listdir(dataset_path):
            if filename.endswith(".png"):
                img_path = os.path.join(dataset_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(Image(img))
                else:
                    print(f"Failed to load image: {img_path}")

        print("Images loaded in objects: ", len(images))
    

        for img in images:
            imagesCollection = self._segmentation_image(img, show, save, conf)
            imagesCollection.save(IMG_SAVE_FILE)

        print("Images saved")


    def _segmentation_image(self, img, save: bool, show: bool, conf: float):
        imgCollection = ImageCollection([])
        for model in self.models:
            results = model.predict(source=img.value, show=show, conf=conf, save=save)

            # crop images with bounding box
            for result in results:
                for boxes in result.boxes:
                    imgCollection.add(img.crop(boxes))

        return imgCollection

    def print(self, string):
        if self.verbose:
            print(string)


if __name__ == "__main__":
    models = []
    models.append(YoloModel(Path("./ia/segmentation/v1.pt")))

    pipeline = Pipeline(models=models, State=PipelineState.ANALYSING)
    pipeline.detect()
