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

class Pipeline:

    def __init__(self, models, unsupervised_model , verbose: bool = True, State: PipelineState= PipelineState.INIT):
        self.verbose = verbose

        self.print("=== Init Pipeline ===")  # Fixed this line

        self.models = []
        for model in models:
            self.models.append(model)
        
        self.unsupervised_model = unsupervised_model

        self._state = State
        if self._state == PipelineState.TRAINING:
            #self._dataManager = Mock_DataManager(Path("./dataset/mock"))
            pass
        else:
            self._dataManager = DataManager(
                "", "./src/cameras.yaml", self.verbose
            ).get_instance()

        self._trainingManager = TrainingManager(is_time_threshold=False, verbose=self.verbose)

    def get_dataset(self) -> None:
        """Génère un dataset avec tout les caméras instancié lors du init du pipeline.

        Utiliser ENTER pour prendre une photo
        Utiliser BACKSPACE pour sortir de la boucle

        Photo sauvegarder dans le dossier dataset

        Return None
        """
        self._state = PipelineState.DATASET
        counter = 0

        for i in range(1000):
            session_path = f"./dataset/session_{i}/"
            if not os.path.exists(session_path):
                os.makedirs(session_path)
                break

        while True:
            key = input("Press 'q' to capture photo, 'e' to exit: ")

            if key == "q":
                Images = self._dataManager.get_all_img()
                if isinstance(Images, list):
                    for i, Image in enumerate(Images):
                        Image.save(
                            os.path.join(
                                session_path, f"photo_camera_{counter}_{i  }.png"
                            )
                        )
                    counter += 1
                else:
                    Image.save(
                        os.path.join(session_path, f"photo_camera_{counter}_{0}.png")
                    )
                    counter += 1
                print("Capture Done")

            if key == "e":
                print("Exit Capture")
                break

        self._state = PipelineState.INIT

    def train(self, yaml_path: str, yolo_model: str, **kargs):

        model = YoloModel(f"{yolo_model}.pt")
        args = dict(data=yaml_path, **kargs)

        if self._state == PipelineState.TRAINING:
            import clearml
            clearml.browser_login()

            task = clearml.Task.init(project_name="PMC", task_name=f"{yolo_model} task")
            task.set_parameter("model_variant", yolo_model)
            task.connect(args)

        results = model.train(**args)

        return results

    def detect(self, show: bool = False, save: bool = False, conf: float = 0.7):
        while True:
            # TODO Utiliser l'API de Mathieu
            key = None
            if self._state != PipelineState.TRAINING:
                key = input("Press 'q' to detect on cameras, 'e' to exit: ")

            if key == "q" or self._state == PipelineState.TRAINING:
                images = self._get_images()
                for img in images:
                    imagesCollection = self._segmentation_image(img, show, save, conf)

                    # TODO Integrate non supervised model

                    # TODO Integrate supervised model

                    # Integrate save
                    imagesCollection.save(IMG_SAVE_FILE)

                    # Integrate training loop
                    if self._trainingManager.check_flags():
                        self._trainingManager.separate_dataset()
                        model = self._trainingManager.train_supervised()

                    # TODO Integrate Classification

                    # TODO send to interface

            if key == "e":
                print("Exit Capture")
                break
        self._state = PipelineState.INIT

    def _get_images(self):
        return self._dataManager.get_all_img()

    def _segmentation_image(self, img: Image, save: bool, show: bool, conf: float):
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
    data_path = './datasets/v8i.yolov8/data.yaml'
    
    pipeline = Pipeline(models=[], unsupervised_model=None, State=PipelineState.TRAINING)

    pipeline.train(yaml_path=data_path, yolo_model="yolov8m-seg", epochs=350, batch=-1, workers=4)

    # data_path = "D:\dataset\dofa_3"

    #data_path = "D:\dataset\dofa_2\data.yaml"
    
    # test_model = YoloModel()
    # test_model.train(epochs=3, data=data_path, batch=-1)

    # test_resultats = test_model.eval()

    # welding_resultats = welding_model.eval()

    # if test_resultats.fitness > welding_resultats.fitness:
    #     print('wrong')

    # print(f'test fitness: {test_resultats.fitness}')
    # print(f'welding fitness: {welding_resultats.fitness}')

    #Pipeline.train(data_path, "yolov8m-cls", epochs=350, batch=15, workers=4)
    #Pipeline = Pipeline(models, training=True)
    #Pipeline.detect()

    # Pipeline.train(data_path, "yolov8n-seg", epochs=350, batch=15, workers=4)

    # Pipeline.get_dataset()

    # import torch

    # if torch.cuda.is_available():
    # model.predict(source="C:\Users\Charles\Pictures\Camera Roll\WIN_20240206_12_40_26_Pro.mp4", show=True, save=True, conf=0.5, device='gpu')

    # Hyperparameter optimizer
    # model = YOLO('yolov8n-seg.pt')
    # model.tune(data=data_path, epochs=30, iterations=20, val=False, batch=-1)

    # Model Training
    # Pipeline.train(data_path, 'yolov8s-seg', epochs=250, plots=False)

    Pipeline.detect()
