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

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Pipeline:
    
    def __init__(self, models, unsupervised_model, verbose: bool = True):

        self.verbose = verbose

        print("=== Init Pipeline ===")  # Fixed this line

        self.models = []
        for model in models:
            self.models.append(model)
        
        self._state = PipelineState.INIT
        self._dataManager = DataManager(
            "", "./Code/src/cameras.yaml", self.verbose
        ).get_instance()
        self.unsupervised_model = unsupervised_model


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
                                session_path, f"photo_camera_{counter}_{i}.png"
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

    def detect(self, show: bool = False, save: bool = True, conf: float = 0.5):
        self._state = PipelineState.ANALYSING
        while True:
            key = input("Press 'q' to detect on cameras, 'e' to exit: ")

            if key == "q":
                Images = self._dataManager.get_all_img()
                for img in Images:
                    for model in self.models:
                        results = model.predict(
                            source=img.value,
                            show=show,
                            save=save,
                            conf=conf,
                            save_crop=True,
                        )

                        # crop images with bounding box
                        cropped_imgs = []
                        for result in results:
                            for boxes in result.boxes:
                                cropped_imgs.append(img.crop(boxes))
                        
                        print("Nb cropped img", len(cropped_imgs))
                               
                
                        for image in cropped_imgs:
                            if True: #C'est un if le model non supervise prend des images subdivises
                                # Calculate the closest multiples for both dimensions
                                closest_width = int(np.ceil(image.value.shape[1] / unsupervised_model.input_shape[1]) * unsupervised_model.input_shape[1])
                                closest_height = int(np.ceil(image.value.shape[0] / unsupervised_model.input_shape[2]) * unsupervised_model.input_shape[2])


                                image.value = cv2.resize(image.value, (closest_width, closest_height))
                                cv2.imshow('Resized Image',image.value)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()

                                sub_images = image.subdivise(unsupervised_model.input_shape[1], 0, "untranslated")

                                for i, sub_image in enumerate(sub_images):
                                    cv2.imshow('Sub image', sub_image.value)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()

                                    sub_image.value = sub_image.value.astype('float32') /255

                                    worst_ssim, worst_ssim_position, worst_ssim_square, worst_ssim_prediction = self.detect_default(sub_image.value, 32, i)

                                    if True:
                                    
                                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                                        # Plot the first RGB image
                                        axes[0].imshow(sub_image.value)
                                        axes[0].set_title(f"Error in image {i}")
                                        axes[0].axis('off')

                                        # Plot the second grayscale image
                                        axes[1].imshow(worst_ssim_square)
                                        axes[1].set_title(f"At position x={worst_ssim_position[0]} y={worst_ssim_position[1]}")
                                        axes[1].axis('off')

                                        # Plot the third grayscale image
                                        axes[2].imshow(worst_ssim_prediction)
                                        axes[2].set_title(f"Prediction made")
                                        axes[2].axis('off')

                                        # Show the combined plot
                                        plt.show()


            if key == 'e':
                print('Exit Capture')
                break
        self._state = PipelineState.INIT
    
    def detect_default(self, sub_image, square_size, image_numb, debug=False):
        # Initialize an array to store the transformed images
        transformed_images = []

        # Get the dimensions of the original image
        rows, cols, channels = sub_image.shape

        # Calculate the number of squares in each dimension
        num_squares_rows = rows // square_size
        num_squares_cols = cols // square_size

        worst_ssim = 1
        worst_ssim_position = [0, 0]
        worst_ssim_square = []
        worst_ssim_prediction = []

        # Iterate through each position
        for j in range(num_squares_cols):
            for i in range(num_squares_rows):
                # Create a copy of the original image
                current_image = np.copy(sub_image)

                # Calculate the coordinates of the square
                square_top_left = (i * square_size, j * square_size)
                square_bottom_right = (square_top_left[0] + square_size, square_top_left[1] + square_size)

                # Black out the square for each channel
                for channel in range(channels):
                    current_image[square_top_left[0]:square_bottom_right[0], square_top_left[1]:square_bottom_right[1], channel] = 0

                cv2.imshow('To be predicted', current_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                current_image_reshaped = current_image.reshape((-1, rows, cols, channels))

                # Generate image
                prediction = unsupervised_model.predict(current_image_reshaped)
                prediction = prediction.reshape(rows, cols, channels)

                cv2.imshow('To be predicted', prediction)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Test to compare
                #prediction_unmodified = model.predict(image.reshape((-1, image_size, image_size, channels)))
                #prediction_unmodified = prediction_unmodified.reshape(image_size, image_size, channels)

                # Compare the blacked out square for each channel separately
                ssim_values = []
                for channel in range(channels):
                    img1_channel = sub_image[square_top_left[0]:square_bottom_right[0], square_top_left[1]:square_bottom_right[1], channel]
                    #img1_channel = prediction_unmodified[:,:,channel]
                    img2_channel = prediction[square_top_left[0]:square_bottom_right[0], square_top_left[1]:square_bottom_right[1], channel]
                    #ssim_index, _ = ssim(img1_channel, img2_channel, full=True, data_range=1)
                    #ssim_values.append(ssim_index)
                    ssim_values.append(np.mean(img2_channel)/np.mean(img1_channel)*100)

                # Average the SSIM values across channels
                avg_ssim = np.mean(ssim_values)
                print(avg_ssim, "%")
                if avg_ssim > 1:
                    print("entered")
                    if avg_ssim > worst_ssim:
                        worst_ssim = avg_ssim
                        worst_ssim_position = [i, j]
                        worst_ssim_square = current_image
                        worst_ssim_prediction = prediction

                    if debug:
                        print(f"Flag in image {image_numb} at square {i} : {j} = {avg_ssim}")

                        # Create subplots for predicted images
                        plt.imshow(sub_image)
                        plt.title(f"Original Image {image_numb}")
                        plt.axis('off')
                        plt.show()

                        # Create subplots for predicted images
                        plt.imshow(prediction)
                        plt.title(f"Predicted Image {image_numb}")
                        plt.axis('off')
                        plt.show()

                        # Create subplots for comparison
                        plt.imshow(current_image)
                        plt.title(f"Original comparison at square {i} : {j}")
                        plt.axis('off')
                        plt.show()

                        # Create subplots for comparison
                        plt.imshow(sub_image)
                        plt.title(f"Predicted comparison at square {i} : {j}")
                        plt.axis('off')
                        plt.show()

        return worst_ssim, worst_ssim_position, worst_ssim_square, worst_ssim_prediction
        

    def print(self, string):
        if self.verbose:
            print(string)


if __name__ == "__main__":
    models = []
    models.append(YoloModel('./Code/src/ia/welding_detection_v1.pt'))
    models.append(YoloModel('./Code/src/ia/piece_detection_v1.pt'))
    unsupervised_model = tf.keras.models.load_model('./Code/src/ia/wandb_fev12_demo.h5')

    # welding_model = YoloModel('./src/ia/welding_detection_v1.pt')

    data_path = "D:\dataset\dofa_2\data.yaml"
    
    # test_model = YoloModel()
    # test_model.train(epochs=3, data=data_path, batch=-1)

    # test_resultats = test_model.eval()

    # welding_resultats = welding_model.eval()

    # if test_resultats.fitness > welding_resultats.fitness:
    #     print('wrong')

    # print(f'test fitness: {test_resultats.fitness}')
    # print(f'welding fitness: {welding_resultats.fitness}')

    Pipeline = Pipeline(models, unsupervised_model, verbose=True)

    #Pipeline.train(data_path, "yolov8m-cls", epochs=350, batch=15, workers=4)

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
