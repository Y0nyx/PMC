from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import cv2
from skimage import measure
from skimage.metrics import structural_similarity as ssim
from pipeline.models.Model import YoloModel

class RocPipeline:
    """
    This class produces ROC curves for the supervised and unsupervised models.
    ::
    Attributes:
        model (?): Model to test against.
        validation_dataset_path (str): Path to the validation dataset.
        curve_name (str): Name of the plotted and saved ROC curve.
        original_images (?): Images to predict on.
        predicted_images (?): Images that were predicted by the model.
        masked_images (?): Mask on original images where defects are.
    Methods:
        generate_roc: Generates the ROC curve.
    """
    def __init__(self, model, validation_dataset_path, curve_name, origina_images, predicted_images, masked_images):
        #TODO: Revoir la logique et le scope de la classe. Si on fait des predict dedans ou non.
        self._model = model
        self._validation_dataset_path = validation_dataset_path
        self._curve_name = curve_name
        self.origina_images = origina_images
        self.predicted_images = predicted_images
        self.masked_images = masked_images

    def _contains_one(self, array) -> int:
        """
        Checks if the input array contains a 1.
        ::
        Args:
            array [float]: Array to check for ones.
        Returns:
            contains_one (int): 1 if the array contains a 1 and 0 otherwise
        """
        return 1 if np.any(array == 1) else 0

    def _parse_annotations(self):
        """
        Parses image annotations in annotation_dir.
        ::
        Args:
        Returns:
            y_true [int]: Array with 1 for images with a defect and 0 for images without.
            filenames [str]: Array of the filenames in the directory.
            count_true (int): Number of images with a defect in the directory.
            count_false (int): Number of images without a defect in the directory.
        """
        annotation_dir = self._validation_dataset_path + "/labels"
        y_true = []
        count_true = 0
        count_false = 0
        files_path = sorted(os.listdir(annotation_dir))
        filenames = []
        # Iterate over text annotation files
        for txt_file in files_path:
            filenames.append(txt_file.split('.txt')[0])
            if txt_file.endswith('.txt'):
                with open(os.path.join(annotation_dir, txt_file), 'r') as file:
                    lines = file.readlines()
                    # Check if there is an associated bounding box
                    if len(lines) > 0:
                        y_true.append(1) # Class 1 if bounding box is present
                        count_true +=1
                    else:
                        y_true.append(0) # Class 0 otherwise
                        count_false += 1

        return y_true, filenames, count_true, count_false

    def _load_images(self):
        """
        Loads the images from _validation_dataset_path.
        ::
        Args:
        Returns:
            images [Image]: Loaded images
        """
        data_path = self._validation_dataset_path + "/images"
        images = []
        for filename in os.listdir(data_path):
            if filename.endswith(".jpg"):
                img = cv2.imread(f'{data_path}/{filename}', cv2.COLOR_RGB2BGR)
                images.append(img)

        #TODO: Mettre image collection (do we care?)
        return images
    
    def _prediction_based_on_metric_threshold(self, origina_images, predicted_images, metric):
        """
        UNTESTED: Makes a prediction based on specified metric(s).
        ::
        Args:
        Returns:
        """
        #results = self._model.predict(images)
        predictions = []
        match metric:
            case 'psnr':
                predictions = self._calculate_psnr_for_images(origina_images, predicted_images)
            case 'ssim':
                predictions = self._calculate_ssim_for_images(origina_images, predicted_images)

        return predictions

    def _calculate_psnr(self, img1, img2):
        """
        UNTESTED: Calculates the psnr.
        ::
        Args:
        Returns:
        """
        psnr = measure.compare_psnr(img1, img2)
        return psnr

    def _calculate_psnr_for_images(self, images, results):
        """
        UNTESTED: Calculates the psnr for specified images.
        ::
        Args:
        Returns:
        """
        psnr_values = []
        for image, result in zip(images, results):
            psnr = self._calculate_psnr(image, result)
            psnr_values.append(psnr)
        return psnr_values

    def _calculate_ssim(self, img1, img2):
        """
        UNTESTED: Calculates the ssim.
        ::
        Args:
        Returns:
        """
        # Ensure images are the same size
        if img1.shape != img2.shape:
            raise ValueError("Images must be the same size")

        # Calculate SSIM
        ssim_value = ssim(img1, img2, multichannel=True)
        return ssim_value

    def _calculate_ssim_for_images(self, images, results):
        """
        UNTESTED: Calculates the ssim for specified images.
        ::
        Args:
        Returns:
        """
        ssim_values = []
        for image, result in zip(images, results):
            ssim_value = self._calculate_ssim(image, result)
            ssim_values.append(ssim_value)
        return ssim_values

    def generate_roc(self, save:bool = True) -> None:
        """
        Generates the roc curve, given the objects attributes.
        ::
        Args:
            save (bool
        Returns:
        """
        y_labels, filenames, count_true, count_false = self._parse_annotations()

        # Trouver les indices des images avec et sans défaut
        indices_true = [i for i, label in enumerate(y_labels) if label == True]
        indices_false = [i for i, label in enumerate(y_labels) if label == False]

        # Sélectionner un nombre égal d'indices aléatoires pour chaque classe
        min_count = min(count_true, count_false)
        selected_indices_true = random.sample(indices_true, min_count)
        selected_indices_false = random.sample(indices_false, min_count)

        # Concaténer les indices sélectionnés
        selected_indices = selected_indices_true + selected_indices_false

        # Utiliser les indices sélectionnés pour obtenir les nouveaux y_labels et filenames équilibrés
        y_labels_balanced = [y_labels[i] for i in selected_indices]
        filenames_balanced = [filenames[i] for i in selected_indices]

        if not self.is_yolo:
            images = self._load_images()
            y_scores = self._prediction_based_on_metric_threshold(images, 'psnr')
        else:
            y_scores = []
            images_path = self._validation_dataset_path + "\\images"
            for img in filenames_balanced:
                results = self._model.predict(source=os.path.join(images_path, img + '.jpg'))
                for result in results:
                    if len(result.boxes) > 0:
                        test = []
                        for box in result.boxes:
                            test.append(box.conf.cpu()[0])
                        y_scores.append(max(test)) 
                    else:
                        y_scores.append(0)

        y_scores = self._prediction_based_on_metric_threshold(images, 'psnr')
        y_labels = self._contains_one(self.masked_images)

        # Calculate true positive rate (TPR) and false positive rate (FPR)
        fpr, tpr, thresholds = roc_curve(y_labels_balanced, y_scores)

        # Calculate the area under the ROC curve (AUC)
        roc_auc = auc(fpr, tpr)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = 1-thresholds[optimal_idx]

        print("Seuil optimal:", optimal_threshold)

        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (ASC = {roc_auc:.2f}), Seuil optimal: {optimal_threshold:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot([0, 0], [1,0],[1,1], color='green', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux faux positifs')
        plt.ylabel('Taux vrai positifs')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right")
        # Save the plot without displaying it
        plt.show()
        plt.savefig('roc_' + self._curve_name + '.png')
        plt.close()

if __name__ == "__main__":
    model = YoloModel('default_detection.pt')

    pip = RocPipeline(model=model, validation_dataset_path="D:\\dataset\\default-detection-format-v2\\valid", curve_name='supervisée')

    pip.generate_roc()
