import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

class UnSupervisedPipeline:

    def __init__(self, model, image):
        self._model = model
        self._image = image
        self._subdivisions = []
        self._nb_x_sub = 0
        self._nb_y_sub = 0
        self._model_shape = model.input_shape
        self._square_size = 32
    
    def set_image(self, image):
        self._image = image

    def resize(self):
        closest_width = int(np.ceil(image.value.shape[1] / unsupervised_model.input_shape[1]) * unsupervised_model.input_shape[1])
        closest_height = int(np.ceil(image.value.shape[0] / unsupervised_model.input_shape[2]) * unsupervised_model.input_shape[2])
        image.value = cv2.resize(image.value, (closest_width, closest_height))
    
    def mask(self, sub_image, x, y):
        # Create a copy of the original image
        current_image = np.copy(sub_image)
        ig1, ig2, channels = current_image.shape
        # Calculate the coordinates of the square
        square_top_left = (x * self._square_size, y * self._square_size)
        square_bottom_right = (square_top_left[0] + self._square_size, square_top_left[1] + self._square_size)
        # Black out the square for each channel
        for channel in range(channels):
            current_image[square_top_left[0]:square_bottom_right[0], square_top_left[1]:square_bottom_right[1], channel] = 0
        
        return current_image

    def debug(self):
        for image in cropped_imgs:
            if True: #C'est un if le model non supervise prend des images subdivises
                # Calculate the closest multiples for both dimensions
                sub_images = image.subdivise(128, 0, "untranslated")

                for i, sub_image in enumerate(sub_images):
                    cv2.imshow('Sub image', sub_image.value)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows
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
                        plt.show
    
    def ssim(self, img1, img2):
        ig1, ig2, channels = img1.shape
        for channel in range(channels):
            img1_channel = img1[:,:, channel]

            img2_channel = img2[:,:, channel]

            ssim_index, _ = ssim(img1_channel, img2_channel, full=True, data_range=1)

            return ssim_index



    def brightness(self, img1, img2):
        ig1, ig2, channels = img1.shape
        for channel in range(channels):
            img1_channel = img1[:,:, channel]
            img2_channel = img2[:,:, channel]

            return (np.mean(img1_channel)/np.mean(img2_channel)*100)


    def comparision(self, img1, img2):
        ssim = self.ssim(img1, img2)
        brightness = self.brightness(img1, img2)

        return 10
    
    def subdivise(self):
        sub_images = []
        # Get the dimensions of the original image
        width, height, channels = self._image.shape

        # Calculate the number of sub-images in both dimensions
        self._nb_x_sub = width // self._model.input_shape[1]
        self._nb_y_sub = height // self._model.input_shape[2]

        # Iterate over the sub-images and save each one with overlap
        for i in range(self._nb_x_sub):
            for j in range(self._nb_y_sub):
                left = i * self._model.input_shape[1]
                top = j * self._model.input_shape[2]
                right = left + self._model.input_shape[1]
                bottom = top + self._model.input_shape[2]

                # TODO: Add overlap code
                # left, top, right, bottom = add_overlap(left, top, right, bottom, width, height, overlap_size)

                # Crop the sub-image using NumPy array slicing
                sub_images.append(self._image[left:right, top:bottom, :])

        self._subdivisions = sub_images
    
    def decision(self, sub_image, prediction):
        comparision = self.comparision(sub_image, prediction)
        return False

    def mask_and_predict(self, sub_image):

        width, height, channels = sub_image.shape
        nb_x_mask = width//self._square_size
        nb_y_mask = height//self._square_size

        for x in range(nb_x_mask):
            for y in range(nb_y_mask):
                masked_sub_image = self.mask(sub_image, x, y)
                cv2.imshow("Image", masked_sub_image)
                # Wait for a key press and then close the window
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                masked_sub_image = masked_sub_image.reshape(1, 128, 128, 3)
                #masked_sub_image = masked_sub_image/255.0
                prediction = self._model.predict(masked_sub_image)
                # Remove singleton dimension and convert prediction to uint8 image format
                prediction_image = np.squeeze(prediction) * 255
                prediction_image = prediction_image.astype(np.uint8)
                cv2.imshow("Image", prediction_image)
                # Wait for a key press and then close the window
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                error = self.decision(sub_image, prediction_image)


        
    def detect_default(self, debug=False):

        self.subdivise()

        # Iterate through each subdivision
        for x in range(self._nb_x_sub):
            for y in range(self._nb_y_sub):
                self.mask_and_predict(self._subdivisions[x*y])