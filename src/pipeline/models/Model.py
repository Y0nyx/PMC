import yaml
from ultralytics import YOLO
from common.Constants import *

class Model:
    def __init__(self, model):
        self._model = model

    def train(self):
        raise NotImplementedError("This function needs to be implemented")

    def eval(self):
        raise NotImplementedError("This function needs to be implemented")

    def predict(self):
        raise NotImplementedError("This function needs to be implemented")


class YoloModel(Model):
    results = None
    def __init__(self, model: str = "yolov8n.pt"):
        super().__init__(YOLO(model))

    def train(self, **kwargs):
        self.results = self._model.train(**kwargs)

    def predict(self, **kwargs):
        return self._model(**kwargs)

    def eval(self):
        self.results = self._model.val()
        return self.results
    
    def generate_yaml(self) -> bool:
        """
        Generate YAML file for training configuration.
        
        Returns:
        - bool: True if YAML generation is successful, False otherwise.
        """
        data = {
            'train': str(TRAIN_FILE.absolute()),
            'val': str(VALID_FILE.absolute()),
            'nc': NC,
            'names': CLASSES
        }

        try:
            with open(YAML_FILE, 'w') as file:
                yaml.dump(data, file)
        except Exception as e:
            print(e)
            return False
        return True

