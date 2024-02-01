from pipeline.models.Model import YoloModel
from pipeline.data.DataManager import DataManager
from common.enums.PipelineStates import PipelineState

import os
import cv2
    
class Pipeline:
    def __init__(self):
        self._state = PipelineState.INIT
        self._dataManager = DataManager("", "./src/cameras.yaml").get_instance()

        self._boundingBoxTracker = BoundingBoxTracker()
        self.piece_detection_model = YoloModel('D:/APP/PMC/repos/src/piece_detection.pt')
    
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
            key = cv2.waitKey(0)

            if key == 13:  # Touche "Enter"
                Images = self._dataManager.get_all_img()
                for i, Image in enumerate(Images):
                    Image.save(os.path.join(session_path, f'photo_camera_{counter}_{i}.png'))
                counter += 1

            elif key == 8:  # Touche "Backspace"
                break
        
        self._state = PipelineState.INIT
        cv2.destroyAllWindows()
    
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

            if key == 13:  # Touche "Enter"
                Images = self._dataManager.get_all_img()
                self.piece_detection_model.predict(Images)

            elif key == 8:  # Touche "Backspace"
                break
        
        self._state = PipelineState.INIT
        cv2.destroyAllWindows()

class BoundingBoxTracker:
    def __init__(self, iou_threshold=0.5):
        self.prev_bounding_boxes = []
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1, box2):
        # Calcule l'Intersection over Union (IoU) entre deux bounding boxes
        x1, y1, w1, h1, p1 = box1
        x2, y2, w2, h2, p2 = box2

        intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        intersection_area = intersection_x * intersection_y
        union_area = w1 * h1 + w2 * h2 - intersection_area

        iou = intersection_area / max(union_area, 1e-10)  # éviter la division par zéro

        return iou

    def is_similar(self, box1, box2):
        # Retourne True si les bounding boxes sont similaires en utilisant IoU
        iou = self.calculate_iou(box1, box2)
        return iou >= self.iou_threshold

    def process_prediction(self, new_bounding_box) -> bool:
        similar_count = 0

        for prev_box in self.prev_bounding_boxes:
            if self.is_similar(new_bounding_box, prev_box):
                similar_count += 1

        self.prev_bounding_boxes.append(new_bounding_box)

        if len(self.prev_bounding_boxes) > 5:
            self.prev_bounding_boxes.pop(0)

        if similar_count >= 4:
            return True
        else:
            return False




if __name__ == "__main__":
    Pipeline = Pipeline()

    #Pipeline.detect_piece()

    Pipeline.get_dataset()
