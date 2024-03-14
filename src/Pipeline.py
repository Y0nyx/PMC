from pipeline.models.Model import YoloModel
from pipeline.data.DataManager import DataManager
from common.enums.PipelineStates import PipelineState
from common.image.Image import Image

import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image as Img
import numpy as np

from ultralytics import YOLO
#from clearml import Task
from common.image.ImageCollection import ImageCollection
from common.utils import DataManager as Mock_DataManager
from common.Constants import *
from TrainingManager import TrainingManager
from pipeline.models.UnSupervisedPipeline import UnSupervisedPipeline

import os
from pathlib import Path
import socket
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class Pipeline:

    def __init__(self, supervised_models, unsupervised_models, verbose: bool = True, State: PipelineState= PipelineState.INIT):
        self.verbose = verbose

        self.print("=== Init Pipeline ===")  # Fixed this line

        self.supervised_models = []
        for model in supervised_models:
            self.supervised_models.append(model)

        self.unsupervised_models = []
        for model in unsupervised_models:
            self.unsupervised_models.append(model)

        self._state = State
        if self._state == PipelineState.TRAINING:
            self._dataManager = Mock_DataManager(Path("./dataset/mock"))
        else:
            self._dataManager = DataManager(
                "", "./cameras.yaml", self.verbose
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

    def train(self, yaml_path: str, yolo_model: YoloModel, **kargs):

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

    def save_images(self, images, folder_path):
        # Get the number of images already present in the folder
        existing_images = [file for file in os.listdir(folder_path) if file.startswith("captured_image_cam")]
        num_existing_images = len(existing_images)

        # Iterate over the images and save them with the appropriate index
        for i, img in enumerate(images):
            index = num_existing_images//5 + 1  # Adjust index based on existing images
            filename = os.path.join(folder_path, f"captured_image_cam_{i}_{index}.png")
            cv2.imwrite(filename, img.value)
            print(f"Capture saved as {filename}")

    def detect(self, show: bool = False, save: bool = False, conf: float = 0.7, cam_debug=False):
        if self._state != PipelineState.TRAINING: self._state = PipelineState.TRAINING

        images = self._get_images()

        self.save_images(images, "../../Datasets/Pipeline_captures/")

        for img in images:
            imagesCollection = self._segmentation_image(img, show, save, conf)
            # TODO Integrate non supervised model
            # TODO Integrate supervised model
            # Integrate save
            #imagesCollection.save(IMG_SAVE_FILE)
            # Integrate training loop
            #if self._trainingManager.check_flags():
                #self._trainingManager.separate_dataset()
                #model = self._trainingManager.train_supervised()
            # TODO Integrate Classification
            # TODO send to interface


        self._state = PipelineState.INIT

    def _get_images(self):
        return self._dataManager.get_all_img()

    def _segmentation_image(self, img: Image, save: bool, show: bool, conf: float):
        imgCollection = ImageCollection([])
        for model in self.supervised_models:
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
        # Replace 'localhost' with the IP address of the machine running the Electron app if they are not on the same machine
        HOST = '127.0.0.1' # The server's hostname or IP address
        PORT = 8002 # The port used by the server
        data_to_send = {'code': 'stop', 'data': 'object data'}
        finished_code = {'code': 'finished'}

        supervised_models = []
        supervised_models.append(YoloModel(Path("./ia/segmentation/v1.pt")))
    
        #unsupervised_models = []
        #unsupervised_models.append(load_model("./ia/unsupervised/default_param_model.keras"))
    
        pipeline = Pipeline(supervised_models=supervised_models, unsupervised_models=[])

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            serialized_data = json.dumps(data_to_send).encode()
            s.send(serialized_data)
            print("Successfully connected to Electron")

            while True: 
                data = s.recv(1024)
                print('Received:', data.decode())

                received_json = json.loads(data.decode())
                if received_json['code'] == "start":
                    print("received start")
                    pipeline.detect(cam_debug=True)
            
                serialized_data = json.dumps(finished_code).encode()
                s.send(serialized_data)
    
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
    
        #Pipeline = Pipeline(models, verbose=True)
    
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
    
        #Pipeline.detect()
    