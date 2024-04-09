from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from skimage import measure
from skimage.metrics import structural_similarity as ssim
from pipeline.models.Model import YoloModel

class RocPipeline:

    def __init__(self, model, validation_dataset_path, curve_name: str):
        if isinstance(model, YoloModel):
            self.is_yolo = True
        self._model = model
        self._validation_dataset_path = validation_dataset_path
        self._curve_name = curve_name

    def parse_annotations(self):
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

    def load_images(self):
        data_path = self._validation_dataset_path + "/images"
        images = []
        for filename in os.listdir(data_path):
            if filename.endswith(".jpg"):
                img = cv2.imread(f'{data_path}/{filename}', cv2.COLOR_RGB2BGR)
                images.append(img)
        
        return images
    
    def prediction_based_on_metric_threshold(self, images, metric):
        results = self._model.predict(images)
        predictions = []
        match metric:
            case 'psnr':
                predictions = self.calculate_psnr_for_images(images, results)
            case 'ssim':
                predictions = self.calculate_ssim_for_images(images, results)

        return predictions

    def calculate_psnr(self, img1, img2):
        psnr = measure.compare_psnr(img1, img2)
        return psnr

    def calculate_psnr_for_images(self, images, results):
        psnr_values = []
        for image, result in zip(images, results):
            psnr = self.calculate_psnr(image, result)
            psnr_values.append(psnr)
        return psnr_values

    def calculate_ssim(self, img1, img2):
        # Ensure images are the same size
        if img1.shape != img2.shape:
            raise ValueError("Images must be the same size")

        # Calculate SSIM
        ssim_value = ssim(img1, img2, multichannel=True)
        return ssim_value

    def calculate_ssim_for_images(self, images, results):
        ssim_values = []
        for image, result in zip(images, results):
            ssim_value = self.calculate_ssim(image, result)
            ssim_values.append(ssim_value)
        return ssim_values

    def generate_roc(self, save = True):
        y_labels, filenames, count_true, count_false = self.parse_annotations()

        if not self.is_yolo:
            images = self.load_images()
            y_scores = self.prediction_based_on_metric_threshold(images, 'psnr')
        else:
            y_scores = []
            images_path = self._validation_dataset_path + "\\images"
            for img in filenames:
                results = self._model.predict(source=os.path.join(images_path, img + '.jpg'))
                for result in results:
                    if len(result.boxes) > 0:
                        test = []
                        for box in result.boxes:
                            test.append(box.conf.cpu()[0])
                        y_scores.append(max(test)) 
                    else:
                        y_scores.append(0)


        # Calculate true positive rate (TPR) and false positive rate (FPR)
        fpr, tpr, thresholds = roc_curve(y_labels, y_scores)

        # Calculate the area under the ROC curve (AUC)
        roc_auc = auc(fpr, tpr)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = 1-thresholds[optimal_idx]

        print("Optimal Threshold:", optimal_threshold)

        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}), Optimal Threshold: {optimal_threshold:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot([0, 0], [1,0],[1,1], color='green', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        # Save the plot without displaying it
        plt.show()
        plt.savefig('roc_' + self._curve_name + '.png')
        plt.close()

if __name__ == "__main__":
    model = YoloModel('default_detection.pt')

    pip = RocPipeline(model=model, validation_dataset_path="D:\\dataset\\default-detection-format-v2\\valid", curve_name='supervisée')

    pip.generate_roc()