import cv2
import numpy as np
#from skimage.metrics import structural_similarity as ssim
from common.image.ImageCollection import ImageCollection
from common.image.Image import Image
from skimage.metrics import structural_similarity as ssim

class UnSupervisedPipeline:

    def __init__(self, model, debug: bool=False):
        self._model = model
        self._subdivisions = []
        self._nb_x_sub = 0
        self._nb_y_sub = 0
        self._model_shape = model.input_shape
        self._square_size = 32
        self.debug = debug
        self.threshold = 0
        self.threshold_algo = "mse, ssim, psnr"

    def resize(self, image):
        closest_width = int(np.ceil(image.value.shape[1] / self._model_shape[1]) * self._model_shape[1])
        closest_height = int(np.ceil(image.value.shape[0] / self._model_shape[2]) * self._model_shape[2])
        image.value = cv2.resize(image.value, (closest_width, closest_height))
        image.mask = cv2.resize(image.mask, (closest_width, closest_height))

        return image
    
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
    
    def subdivise(self, image):
        image = self.resize(image)
        sub_images = []
        sub_masks = []
        # Get the dimensions of the original image
        width, height, channels = image.value.shape

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
                sub_images.append(image.value[left:right, top:bottom, :])
                sub_masks.append(image.mask[left:right, top:bottom, :])
        return sub_images, sub_masks
    
    def decision(self, sub_image, prediction):
        comparision = self.comparision(sub_image, prediction)
        return False

    def mask_and_predict(self, sub_image):

        imgCollection = ImageCollection([])
        threshold_results = []
        csv_rows = []

        width, height, channels = sub_image.shape
        nb_x_mask = width//self._square_size
        nb_y_mask = height//self._square_size

        total_images = 0
        average_predicted_image = np.zeros_like(sub_image, dtype=np.float32)

        for x in range(nb_x_mask):
            for y in range(nb_y_mask):
                masked_sub_image = self.mask(sub_image, x, y)
                masked_sub_image = self.masked_sub_image_preprocessing(masked_sub_image)

                prediction = self._model.predict(masked_sub_image, verbose=0)
                predicted_image = self.predicted_image_postprocessing(prediction)

                mse, ssim_value, psnr_value, decision  = self.threshold_comparison(sub_image, predicted_image)
                bb = ""
                if decision: bb = self.calculate_bounding_box(x, y)
                #print(bb)
                threshold_results.append([mse, ssim_value, psnr_value, decision, bb])
                self._csv_row.unsup_defect_threshold = self.threshold
                self._csv_row.unsup_threshold_algo = self.threshold_algo
                self._csv_row.unsup_defect_res = decision
                self._csv_row.unsup_defect_bb = bb
                csv_rows.append(self._csv_row)
                predicted_image = Image(predicted_image)
                imgCollection.add(predicted_image)

                # Accumulate predicted image for averaging
                average_predicted_image += predicted_image.asarray().astype(np.float32)
                total_images += 1

        # Calculate average predicted image
        average_predicted_image /= total_images
        average_predicted_image = average_predicted_image.astype(np.uint8)

        return csv_rows, [imgCollection, threshold_results], average_predicted_image

    def masked_sub_image_preprocessing(self, masked_sub_image):
        #TODO: Subdivide in separate functions and add bool for grayscale conversion
        masked_sub_image = cv2.cvtColor(masked_sub_image, cv2.COLOR_BGR2GRAY)
        masked_sub_image = cv2.cvtColor(masked_sub_image, cv2.COLOR_GRAY2RGB)
        masked_sub_image = masked_sub_image.reshape(1, self._model.input_shape[1], self._model.input_shape[2], self._model.input_shape[3])
        
        masked_sub_image = masked_sub_image/255.0

        return masked_sub_image

    def predicted_image_postprocessing(self, prediction):
        #TODO: Subdivied in separate functions
        predicted_image = np.squeeze(prediction) * 255
        predicted_image = predicted_image.astype(np.uint8)

        return predicted_image
    
    def threshold_comparison(self, original_image, prediction):
        decision = False
        # Mean Squared Error
        mse = np.mean((original_image - prediction) ** 2)

        # Structural Similarity Index
        ssim_value = ssim(original_image, prediction, multichannel=True, win_size=3)

        # Peak Signal-to-Noise Ratio
        max_pixel_value = 255.0 # Assuming 8-bit images
        mse_value = mse
        psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse_value))

        #print(f"MSE: {mse}, SSIM: {ssim_value}, PSNR: {psnr_value}")

        #Set threshold here
        if mse > 0 and ssim_value > 0 and psnr_value > 0: decision = True

        return mse, ssim_value, psnr_value, decision
    
    def calculate_bounding_box(self, x, y):
        return f"{x}:{y} - subdivision position"
        
    def detect_defect(self, image, csv_row):

        self._csv_row = csv_row

        sub_images, sub_masks = self.subdivise(image)
        predicted_collection = []
        csv_rows_collection = []

        average_predicted_images = []

        # Iterate through each subdivision
        for x in range(self._nb_x_sub):
            for y in range(self._nb_y_sub):
                # Correctly index into the subdivisions list
                subdivision_index = x * self._nb_y_sub + y
                csv_rows, predictions, average_predicted_image = self.mask_and_predict(sub_images[subdivision_index])
                average_predicted_images.append(average_predicted_image)
                predicted_collection.append(predictions)
                csv_rows_collection.extend(csv_rows)

        return csv_rows_collection, predicted_collection, sub_masks, sub_images, average_predicted_images