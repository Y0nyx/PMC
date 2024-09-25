from datetime import datetime
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow.keras.models import load_model
from PIL import Image as Img
import numpy as np
import asyncio
import os
import threading
from pathlib import Path
import json
import re
#from clearml import Task

from pipeline.models.Model import YoloModel
from pipeline.models.Callbacks import on_fit_epoch_end, on_train_epoch_end
from pipeline.data.DataManager import DataManager
from pipeline.CsvManager import CsvManager
from pipeline.CsvResultRow import CsvResultRow
from pipeline.models.UnSupervisedPipeline import UnSupervisedPipeline

from common.enums.PipelineStates import PipelineState
from common.image.Image import Image
from common.image.ImageCollection import ImageCollection
from common.utils import DataManager as Mock_DataManager
from common.Constants import *

from NetworkManager import NetworkManager
from TrainingManager import TrainingManager
from RocPipeline import RocPipeline

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class Pipeline():
    """
    This class manages the AI detection pipeline, multiple methods allow the use and training of the detection neural networks.
    ::
    Attributes:
        supervised_models (?): List of supervised models used in the pipeline (segmentation model and detection model usually).
        unsupervised_model (?): The unsupervised detection model.
        current_iteration_logging_path (str): path for all the logs of the current iteration.
        verbose (bool): Flag for verbose prints.
        State (PipelineState): State of the pipeline, based on PipelineState struct.
        csv_logging (bool): Flag for csv_logging of results.
        roc_curve (bool): Flag for roc curve generation when doing detection
    Methods:
        start: Starts all services.
    """
    def __init__(self, segmentation_model, supervised_detection_model, unsupervised_model, current_iteration_logging_path: str, verbose: bool = True, State: PipelineState= PipelineState.INIT, csv_logging: bool = False, roc_curve: bool = False) -> None:
        self.verbose = verbose

        self.stop_flag = threading.Event()

        #Init network manager
        self.network_manager = NetworkManager(HOST, PORT, SUPERVISED_HOST, SUPERVISED_PORT, UNSUPERVISED_HOST, UNSUPERVISED_PORT, self.verbose)

        self.print("=== Init Pipeline ===")  # Fixed this line

        #Set supervised models
        self.supervised_detection_model = supervised_detection_model

        self.segmentation_model = segmentation_model

        #Set unsupervised model
        self.unsupervised_model = unsupervised_model

        self._current_iteration_logging_path = current_iteration_logging_path

        #Set initial pipeline state
        self._state = State
        if self._state == PipelineState.TRAINING:
            self._dataManager = Mock_DataManager(Path("./dataset/mock"))
            pass
        else:
            self._dataManager = DataManager(
                "", "./cameras.yaml", self.verbose
            ).get_instance()

        #Init the training manager
        self._trainingManager = TrainingManager(is_time_threshold=False, verbose=self.verbose)

        self._csv_logging = csv_logging
        if roc_curve:
            self.roc_pipeline = RocPipeline(None, "Test", "unsupervised_test_roc")
        if csv_logging:
            # Init and set CSV logging tools

            self.csv_manager = CsvManager()
            self._csv_result_row = CsvResultRow()

            self._csv_result_row.un_sup_model_ref = UNSUPERVISED_MODEL_REF
            self._csv_result_row.seg_model_ref = SEGMENTATION_MODEL_REF
            self._csv_result_row.sup_model_ref = SUPERVISED_DETECTION_MODEL_REF

    def start(self):
        """
        Method called for this object when its thread starts. It starts the detection code
        ::
        Args:
        Returns:
            results (?): Detection results
        """
        #TODO: Rename ou modifier pour avoir un start de détection et de training
        self.print('START SET')
        self.stop_flag.clear()
        results = self.detect()
        self.print('DONE DETECTING')
        return results

    def stop(self):
        """
        Method called for this object when its thread stops.
        ::
        Args:
        Returns:
            None
        """
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

    def train(self, yaml_path: str, yolo_model: str, **kargs):
        #TODO Mettre ça ailleurs
        model = YoloModel(f"{yolo_model}.pt")

        # add important callbacks
        model._model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model._model.add_callback("on_train_epoch_end", on_train_epoch_end)

        args = dict(data=yaml_path, **kargs)

        if self._state == PipelineState.TRAINING:
            import clearml
            clearml.browser_login()

            task = clearml.Task.init(project_name="PMC", task_name=f"{yolo_model} task")
            task.set_parameter("model_variant", yolo_model)
            task.connect(args)

        results = model.train(**args)

        return results

    def detect(self, show: bool = False, save_seg: bool = False, conf: float = 0.7):
        """
        Detection method, it calls all necessary sub methods to take the pictures, segment them and make a prediction on the wether there is a defect or not
        ::
        Args:
            show (bool): Flag for debug purposes, will show the image with matplotlib. (default = False)
            save_seg (bool): Flag to save the segmented images. (default = False)
            conf (float): Confidence level for the segmentation detection. (default = 0.7)
        Returns:
            results (?): Detection results
        """
        if self._state != PipelineState.TRAINING:
            self._state = PipelineState.TRAINING

        captured_image_collection = self._get_images()
        capture_image_collection_base_path = f"{self._current_iteration_logging_path}/"
        print(capture_image_collection_base_path)
        detection_id = self._count_folders(capture_image_collection_base_path)
        self._captured_image_collection_path = f"{capture_image_collection_base_path}detection_{detection_id}"
        
        captured_image_collection.save(self._captured_image_collection_path + "/images/")

        if captured_image_collection.img_count > 0:
            for i, captured_image in enumerate(captured_image_collection):

                #Check if stopped was received
                if not self.stop_flag.is_set():

                    #Find the welds in the capture image
                    segmented_image_collection, boxes = self._segmentation_image(captured_image, i, show, save_seg, conf)
                    # TODO Integrate supervised model
                    
                else:
                    return
                    
            solder_defect = False
            #TODO: envoyer un dossier à la place d'une image, car on a 5 images par piece
            result_data = {
                "code": "resultat",
                "data": {"resultat": solder_defect, "url": self._captured_image_collection_path+"/images/", "boundingbox": self._captured_image_collection_path + f"/bounding_boxes/",'erreurSoudure':'1'}
            }
            self.print("Finished detection")
            return result_data
                
        else:
            self.print("No image found")
            return {
                "code": "resultat",
                "data": {"resultat": solder_defect, "url": self._captured_image_collection_path+"/images/", "boundingbox": "N/A",'erreurSoudure':'0'}
            }
        
    def _count_folders(self, directory):
        folder_count = len([item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))])
        return folder_count

    def _write_yolo_bounding_boxes(self, result, img_width, img_height, output_file):
        """
        Writes bounding boxes from result.boxes in YOLO format to a .txt file.
        Args:
            result: The detection result containing bounding boxes and class IDs.
            img_width: The width of the image.
            img_height: The height of the image.
            output_file: Path to the output .txt file.
        """
        print("yo")
        with open(output_file, 'w') as f:
            print("yo2")
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                f.write(f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                print("yo3")

    def _unsupervised_defect_detection(self, i: int, segmented_image_collection):
        """
        Method to perform detection using the unsupervised model.
        ::
        Args:
            i (int): Index of the segmentation in the segmentation collection.
            segmented_image_collection (?): All the segmented images of welds. 
        Returns:
            results (?): Detection results
        """
        unsupervised_result_collections = []
        sub_mask_collection = []
        sub_image_collection = []
        average_predicted_sub_image_collection = []
        csv_result_rows = []

        for segmentation in segmented_image_collection:
            csv_result_rows, unsupervised_result_collection, sub_masks, sub_images, average_predicted_images = self._unsupervised_pipeline.detect_defect(segmentation, self._csv_result_row)
            
            unsupervised_result_collections.append(unsupervised_result_collection)
            sub_mask_collection.append(sub_masks)
            sub_image_collection.append(sub_images)
            average_predicted_sub_image_collection.append(average_predicted_images)

        if self._csv_logging:            
            for y, unsupervised_result_collection in enumerate(unsupervised_result_collections):
                for z, unsupervised_results in enumerate(unsupervised_result_collection):
                    unsupervised_results_path = f"{SAVE_PATH}{self._current_iteration_logging_path}{SAVE_PATH_UNSUPERVISED_PREDICTION}_{i}{SAVE_PATH_SEGMENTATION}_{y}{SAVE_PATH_SUBDIVISION}_{z}"
                    unsupervised_results[0].save(unsupervised_results_path)

                    self.write_csv_rows(csv_result_rows, unsupervised_results_path)

        return unsupervised_result_collections, sub_mask_collection, sub_image_collection, average_predicted_sub_image_collection

    def _get_images(self):
        """
        Method to get images captured by the cameras.
        ::
        Args:
        Returns:
            images (?): List of images captured by the cameras
        """
        images = self._dataManager.get_all_img()
        
        if self._csv_logging and images.img_count > 0:
            captured_image_path = f"{SAVE_PATH}{self._current_iteration_logging_path}{SAVE_PATH_CAPTURE}"
            images.save(captured_image_path)
            self._csv_result_row.capture_img_path = captured_image_path

        return images

    def _segmentation_image(self, img: Image, img_id: int, save: bool, show: bool, conf: float):
        """
        Gets the segmented weld out of the captured image using the supervised segmentation model.
        ::
        Args:
            img (Image): Image captured by the camera.
            save (bool): Flag to save the segmentation.
            show (bool): Flag to show the segmentation using matplotlib.
            conf (float): Confidence threshold for the segmentation detection .
        Returns:
            imgCollection (?): Collection of all segmentations detection in the image.
        """
        #TODO: Rename la fonction poru get
        imgCollection = ImageCollection()
        model = self.segmentation_model
        results = model.predict(source=img.value, show=show, conf=conf, save=save)
        os.makedirs(self._captured_image_collection_path + f"/bounding_boxes/", exist_ok=True)
        # crop images with bounding box
        for i, result in enumerate(results):
            self._write_yolo_bounding_boxes(result, 640, 640, self._captured_image_collection_path + f"/bounding_boxes/img_{img_id}.txt")
            for boxe in result.boxes:
                image = img.crop(boxe)
                imgCollection.add(image)

        return imgCollection, results
    
    def _supervised_detection(self, imgCol: ImageCollection, save: bool, show: bool, conf: float):
        imgCollection = ImageCollection([])
        model = self._supervised_detection_model

        for img in imgCol:
            results = model.predict(source=img.value, show=False, conf=conf, save=False)
            # crop images with bounding box
            for i, result in enumerate(results):
                for j, boxe in enumerate(result.boxes):
                    print(img.shape)
                    print(boxe)
                    image = img.crop(boxe)
                    image.save(f"{SAVE_PATH}/result_{i}_box_{j}_detection.png")
                    imgCollection.add(image)

    def print(self, string):
        if self.verbose:
            print(string)

    def write_csv_rows(self, csv_rows, unsupervised_results_path) -> None:
        """
        DEPRECATED Writes the csv rows provided.
        ::
        Args:
        Returns:
        """
        for i, csv_row in enumerate(csv_rows):
            csv_row.unsup_pred_img_path = unsupervised_results_path + "_img_" +str(i)
            csv_row.sup_defect_res = ""
            csv_row.sup_defect_threshold = ""
            csv_row.sup_defect_bb = ""
            csv_row.manual_verification_result = ""
            csv_row.date = datetime.now()
            self.csv_manager.add_new_row(self._csv_result_row)
    
    def model_test(self, show: bool = False, save: bool = False, conf: float = 0.7) -> None:
        """
        Gets the segmented weld out of the captured image using the supervised segmentation model.
        ::
        Args:
            show (bool): Flag to show the segmentation using matplotlib.
            save (bool): Flag to save the segmentation.
            conf (float): Confidence threshold for the segmentation detection .
        Returns:
            None
        """
        #TODO: Unfinished?
        if self._state != PipelineState.TRAINING:
            self._state = PipelineState.TRAINING
        
        #TODO: Change to use the provided test dataset
        captured_image_collection = self._get_images()

        if captured_image_collection.img_count > 0:

            for i, captured_image in enumerate(captured_image_collection):
                #TODO: get the bounding box x and y values and image information
                masked_image = self.create_bounding_box_mask(x1, y1, x2, y2, image_width, image_height)

                #Check if stopped was received
                if not self.stop_flag.is_set():

                    #Find the welds in the capture image
                    segmented_image_collection, boxList = self._segmentation_image(captured_image, show, save, conf)
                    segmented_mask_collection = self.crop_masked_images(masked_image, boxList)

def count_folders_starting_with(start_string, path):
    count = 0
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.startswith(start_string):
                count += 1
    return count

def path_initialization():
    """
    Initializes all necessary paths and returns them.
    ::
    Args:
    Returns:
        ?
    """
    #Init path to segmentation model
    segmentation_model_path = f"{SEGMENTATION_MODEL_PATH}{SEGMENTATION_MODEL_REF}.pt"

    #Init path to unsupervised detection model
    unsupervised_model_higher_path = "../../"
    unsupervised_model_path = f'{unsupervised_model_higher_path}Models/{UNSUPERVISED_MODEL_REF}.keras'

    #Init path to current model iteration for logging purposes
    current_iteration_latest_version_id = count_folders_starting_with(f"{UNSUPERVISED_MODEL_REF}", unsupervised_model_higher_path)
    current_iteration_logging_path = f'/{UNSUPERVISED_MODEL_REF}_{current_iteration_latest_version_id}'

    return segmentation_model_path, unsupervised_model_path, current_iteration_logging_path



if __name__ == "__main__":
    #supervised_models = [YoloModel(Path("./ia/segmentation/v1.pt"))]
    # TRAINING
    # pipeline = Pipeline(supervised_models=[], unsupervised_models=[], State=PipelineState.TRAINING)


    # data_path = "../../Datasets/default-detection-format-v3/data.yaml"
    # pipeline.train(data_path, "yolov8l", epochs=350, batch=-1, workers=0)

    segmentation_model_path, unsupervised_model_path, current_iteration_logging_path  = path_initialization()

    supervised_models = [YoloModel(Path(segmentation_model_path))]
    unsupervised_model = tf.keras.models.load_model(unsupervised_model_path)

    pipeline = Pipeline(supervised_models=supervised_models, unsupervised_model=unsupervised_model, current_iteration_logging_path=current_iteration_logging_path, State=PipelineState.TRAINING)

    
