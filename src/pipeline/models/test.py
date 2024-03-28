from UnSupervisedPipeline import UnSupervisedPipeline
import cv2
import tensorflow as tf

image_path = 'C:/Users/mafrc/Desktop/Uni/PMC/CodePMC/runs/segment/predict17/crops/soudure/image0.jpg'

# Read the image using OpenCV
image = cv2.imread(image_path)
cv2.imshow("Image", image)
# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Hello world")

unsupervised_model = tf.keras.models.load_model('./Code/src/ia/wandb_fev12_demo.h5')
un_supervised_pipeline = UnSupervisedPipeline(unsupervised_model, image)
un_supervised_pipeline.detect_defect()
