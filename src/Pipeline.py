from pipeline.models.Model import YoloModel
from pipeline.data.DataManager import DataManager
from NetworkManager import NetworkManager
from common.enums.PipelineStates import PipelineState
from common.image.Image import Image
from pipeline.CsvManager import CsvManager
from pipeline.CsvResultRow import CsvResultRow
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras.models import load_model
from PIL import Image as Img
import numpy as np
import asyncio

#from clearml import Task
from common.image.ImageCollection import ImageCollection
from common.utils import DataManager as Mock_DataManager
from common.Constants import *
from TrainingManager import TrainingManager
from pipeline.models.UnSupervisedPipeline import UnSupervisedPipeline

import os
import threading
from pathlib import Path
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class Pipeline():
    def __init__(self, supervised_models, unsupervised_model, verbose: bool = True, State: PipelineState= PipelineState.INIT, current_iteration_save_folder = ""):
        self.stop_flag = threading.Event()

        self.verbose = verbose
        self.network_manager = NetworkManager(HOST, PORT, self.verbose)

        self.print("=== Init Pipeline ===")  # Fixed this line

        self.supervised_models = []
        for model in supervised_models:
            self.supervised_models.append(model)

        self.unsupervised_model = unsupervised_model

        self._state = State
        if self._state == PipelineState.TRAINING:
            self._dataManager = Mock_DataManager(Path("./dataset/mock"))
        else:
            self._dataManager = DataManager(
                "", "./cameras.yaml", self.verbose
            ).get_instance()

        self._trainingManager = TrainingManager(is_time_threshold=False, verbose=self.verbose)
        self._current_iteration_save_folder = current_iteration_save_folder

        self.csv_manager = CsvManager()
        self._csv_result_row = CsvResultRow()

        self._csv_result_row.un_sup_model_ref = UNSUPERVISED_MODEL_REF
        self._csv_result_row.seg_model_ref = SEGMENTATION_MODEL_REF
        self._csv_result_row.sup_model_ref = SUPERVISED_MODEL_REF

    def start(self):
        self.print('START SET')
        self.stop_flag.clear()
        return self.detect(cam_debug=True)

    def stop(self):
        self.print('STOP SET')
        self.stop_flag.set()

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

    def detect(self, show: bool = False, save: bool = False, conf: float = 0.7, cam_debug=False):
        if self._state != PipelineState.TRAINING:
            self._state = PipelineState.TRAINING

        captured_image_collection = self._get_images()
        if captured_image_collection.img_count > 0:
            
            #TODO: Ajouter une condition debug pour choisir si on save ou pas
            #Save the captured images
            #TODO: Mettre les save dans les fonctions qui génèrent les images
            captured_image_path = f"{SAVE_PATH}{self._current_iteration_save_folder}{SAVE_PATH_CAPTURE}"
            captured_image_collection.save(captured_image_path)
            self._csv_result_row.capture_img_path = captured_image_path

            for i, captured_image in enumerate(captured_image_collection):

                #Check if stopped was received
                if not self.stop_flag.is_set():

                    #Find the welds in the capture image
                    segmented_image_collection = self._segmentation_image(captured_image, show, save, conf)
                    segmented_image_collection_path = f"{SAVE_PATH}{self._current_iteration_save_folder}{SAVE_PATH_SEGMENTATION}"
                    segmented_image_collection.save(segmented_image_collection_path)
                    self._csv_result_row.seg_img_path = segmented_image_collection_path
                    self._csv_result_row.seg_results = segmented_image_collection.img_count
                    self._csv_result_row.seg_threshold = conf

                    #Analyse the welds with the unsupervised model to find potential defaults
                    unsupervised_result_collections = []
                    csv_result_rows = []
                    for segmentation in segmented_image_collection:
                        unsupervised_pipeline = UnSupervisedPipeline(self.unsupervised_model, segmentation, self._csv_result_row)
                        csv_result_rows, unsupervised_result_collection = unsupervised_pipeline.detect_defect()
                        unsupervised_result_collections.append(unsupervised_result_collection)
                    
                    for y, unsupervised_result_collection in enumerate(unsupervised_result_collections):
                        for z, unsupervised_results in enumerate(unsupervised_result_collection):
                            unsupervised_results_path = f"{SAVE_PATH}{self._current_iteration_save_folder}{SAVE_PATH_UNSUPERVISED_PREDICTION}_{i}{SAVE_PATH_SEGMENTATION}_{y}{SAVE_PATH_SUBDIVISION}_{z}"
                            unsupervised_results[0].save(unsupervised_results_path)

                            self.write_csv_rows(csv_result_rows, unsupervised_results_path)




                    # TODO Integrate supervised model
                    

                    # Integrate training loop
                    #if self._trainingManager.check_flags():
                    #    self._trainingManager.separate_dataset()
                    #    model = self._trainingManager.train_supervised()

                    # TODO Integrate Classification

                    # TODO send to interface

                    # Check if a stop signal has been received
                    #if await self.check_stop_signal():
                    #    print("Stop signal received, stopping detection")
                    #    return
                    
                result_data = {
                    "resultat": True,  # or False based on your condition
                    "url": "/imageSoudure....",
                    "erreurSoudure": "pepe"
                }

                # Convert the dictionary to JSON format
                result_json = json.dumps(result_data)

                self._state = PipelineState.INIT

                # Send the JSON data
                #return result_json
            else:
                return
        else:
            self.print("No image found")
            return {
                "resultat": None,  # or False based on your condition
                "url": "/imageSoudure....",
                "erreurSoudure": "pepe"
            }

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

    def write_csv_rows(self, csv_rows, unsupervised_results_path):
        for i, csv_row in enumerate(csv_rows):
            csv_row.unsup_pred_img_path = unsupervised_results_path + "_img_" +str(i)
            csv_row.sup_defect_res = ""
            csv_row.sup_defect_threshold = ""
            csv_row.sup_defect_bb = ""
            csv_row.manual_verification_result = ""
            csv_row.date = datetime.now()
            self.csv_manager.add_new_row(self._csv_result_row)

def count_folders_starting_with(start_string, path):
    count = 0
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.startswith(start_string):
                count += 1
    return count

if __name__ == "__main__":

    supervised_models = [YoloModel(Path(f"./ia/segmentation/{SEGMENTATION_MODEL_REF}.pt"))]

    unsupervised_model_higher_path = "../../"
    unsupervised_model_path = f'{unsupervised_model_higher_path}{UNSUPERVISED_MODEL_REF}.keras'
    unsupervised_model = tf.keras.models.load_model(unsupervised_model_path)
    folder_count = count_folders_starting_with(f"{UNSUPERVISED_MODEL_REF}", unsupervised_model_higher_path)
    current_iteration_save_folder = f'/{UNSUPERVISED_MODEL_REF}_{folder_count}'

    pipeline = Pipeline(supervised_models=supervised_models, unsupervised_model=unsupervised_model, current_iteration_save_folder=current_iteration_save_folder)

    pipeline.detect()
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
    