from pipeline.models.Model import YoloModel
from pipeline.data.DataManager import DataManager
from common.enums.PipelineStates import PipelineState

import os
import cv2

from ultralytics import YOLO
from clearml import Task

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Pipeline:
    def __init__(self):
        self._state = PipelineState.INIT
        self._dataManager = DataManager("", "./src/cameras.yaml").get_instance()
    
    def get_dataset(self) -> None:
        """ Génère un dataset avec tout les caméras instancié lors du init du pipeline.

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

            if key == 'q':
                Images = self._dataManager.get_all_img()
                if isinstance(Images, list):
                    for i, Image in enumerate(Images):
                        Image.save(os.path.join(session_path, f'photo_camera_{counter}_{i}.png'))
                    counter += 1
                else:
                    Image.save(os.path.join(session_path, f'photo_camera_{counter}_{0}.png'))
                    counter += 1
                print('Capture Done')
            
            if key == 'e':
                print('Exit Capture')
                break
        
        self._state = PipelineState.INIT
    
    def detect_piece(self):
        """ Effectuer un détection de pièce avec le modèle choisie.
            peut sauvegarder le crop si nécessaire.

            Utiliser ENTER pour prendre une photo
            Utiliser BACKSPACE pour sortir de la boucle

            Return None
        """
        self._state = PipelineState.ANALYSING

        while True:
            key = cv2.waitKey(0)

            if key == ord('q'):  # Touche "q"
                Images = self._dataManager.get_all_img()
                self.piece_detection_model.predict(Images)

            if key == ord('e'):  # Touche "e"
                break
        
        self._state = PipelineState.INIT
        cv2.destroyAllWindows()
    
    def train(self, yaml_path, yolo_model, kargs):
        task = Task.init(
            project_name="PMC",
            task_name=f"{yolo_model} task"
        )

        task.set_parameter("model_variant", yolo_model)

        model = YoloModel(f'{yolo_model}.pt')

        args = dict(data=yaml_path, **kargs)
        task.connect(args)

        results = model.train(yaml_path, **args)

    def detect(self, pt_file):
        model = YOLO(pt_file)
        model.predict(source='D:\Documents\Test', show=True, save=True, conf=0.5)

if __name__ == "__main__":
    Pipeline = Pipeline()

    #Pipeline.detect_piece()

    Pipeline.get_dataset()

    #data_path = "D:\dataset\dofa_2\data.yaml"

    #import torch

    #if torch.cuda.is_available():
        #model.predict(source="C:\Users\Charles\Pictures\Camera Roll\WIN_20240206_12_40_26_Pro.mp4", show=True, save=True, conf=0.5, device='gpu')

        # Hyperparameter optimizer 
        #model = YOLO('yolov8n-seg.pt')
        #model.tune(data=data_path, epochs=30, iterations=20, val=False, batch=-1)

    # Model Training
    #Pipeline.train(data_path, 'yolov8s-seg', epochs=250, plots=False)

        #Pipeline.detect("D:/APP/PMC/repos/runs/segment/train4/weights/best.pt")
