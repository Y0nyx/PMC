from pipeline.models.Model import YoloModel
from pipeline.data.DataManager import DataManager
from common.enums.PipelineStates import PipelineState
from common.image import ImageCollection, Image
from common.utils import DataManager as Mock_DataManager
from common.Constants import *
from TrainingManager import TrainingManager

import os

from clearml import Task

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Pipeline:
    def __init__(self, models: list = [], verbose: bool = True, training=False):
        self.verbose = verbose
        self.training = training

        self.print("=== Init Pipeline ===")

        self.models = []
        for model in models:
            self.models.append(model)

        self._state = PipelineState.INIT
        if self.training:
            self._dataManager = Mock_DataManager("D:\\APP\\PMC\\repos\\dataset\\mock")
        else:
            self._dataManager = DataManager(
                "", "./src/cameras.yaml", self.verbose
            ).get_instance()

        self._trainingManager = TrainingManager()

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

    def train(self, yaml_path, yolo_model, **kargs):
        task = Task.init(project_name="PMC", task_name=f"{yolo_model} task")

        task.set_parameter("model_variant", yolo_model)

        model = YoloModel(f"{yolo_model}.pt")

        args = dict(data=yaml_path, **kargs)
        task.connect(args)

        results = model.train(**args)

    def detect(self, show: bool = False, save: bool = True, conf: float = 0.7):
        while True:
            key = None
            if not self.training:
                key = input("Press 'q' to detect on cameras, 'e' to exit: ")

            if key == "q" or self.training:
                images = self._get_images()
                for img in images:
                    images = self._process_image(img, show, save, conf)

                    # TODO Integrate non supervised model

                    # TODO Integrate supervised model

                    # Integrate save
                    images.save(IMG_SAVE_FILE)

                    # Integrate training loop
                    self._trainingManager.check_flags()

                    # Integrate save

                    # TODO Integrate Classification

                    # TODO send to interface

            if key == "e":
                print("Exit Capture")
                break
        self._state = PipelineState.INIT

    def _get_images(self):
        return self._dataManager.get_all_img()

    def _process_image(self, img: Image, save: bool, show: bool, conf: float):
        for model in self.models:
            results = model.predict(source=img.value, show=show, conf=conf, save=save)

            # crop images with bounding box
            imageCollection = ImageCollection()
            for result in results:
                for boxes in result.boxes:
                    imageCollection.add(img.crop(boxes))

        return imageCollection

    def print(self, string):
        if self.verbose:
            print(string)


if __name__ == "__main__":
    models = []
    models.append(YoloModel("./src/ia/welding_detection_v1.pt"))
    # models.append(YoloModel('./src/ia/piece_detection_v1.pt'))

    # welding_model = YoloModel('./src/ia/welding_detection_v1.pt')

    # data_path = "D:\dataset\dofa_3"

    # test_model = YoloModel()
    # test_model.train(epochs=3, data=data_path, batch=-1)

    # test_resultats = test_model.eval()

    # welding_resultats = welding_model.eval()

    # if test_resultats.fitness > welding_resultats.fitness:
    #     print('wrong')

    # print(f'test fitness: {test_resultats.fitness}')
    # print(f'welding fitness: {welding_resultats.fitness}')

    Pipeline = Pipeline(models, training=True)
    Pipeline.detect()

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

    # Pipeline.detect("D:/APP/PMC/repos/runs/segment/train4/weights/best.pt")
