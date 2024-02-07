class Model:
    def __init__(self, model):
        self._model = model
    
    def train(self):
        raise NotImplementedError('This function needs to be implemented')
    
    def eval(self):
        raise NotImplementedError('This function needs to be implemented')
    
    def predict(self):
        raise NotImplementedError('This function needs to be implemented')

from ultralytics import YOLO

class YoloModel(Model):
    def __init__(self, model: str = 'yolov8n.pt'):
        super().__init__(YOLO(model))

    def train(self, data_path, args):
        self._model.train(data=data_path, **args)

    def predict(self, images):
        return self._model(images, save=True)

    def eval(self):
        return self._model.val()
