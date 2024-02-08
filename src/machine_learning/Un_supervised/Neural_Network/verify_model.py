import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
from keras.preprocessing import image

image_size = 128
nb_random_images = 3
square_size = 16

def load_model():
    # Load the model
    model_path = 'C:/Users/mafrc/Desktop/Uni/PMC/CodePMC/models/wandb_night_run_basic_CAE_best_gen.h5'
    loaded_model = tf.keras.models.load_model(model_path)

    return loaded_model

def add_horizontal_line(image, line_width, line_length):
    # Add a horizontal white line to the image
    rows, cols = image.shape
    line_top = rows // 2 - line_width // 2
    line_bottom = line_top + line_width
    image[line_top:line_bottom, cols // 2 - line_length // 2:cols // 2 + line_length // 2] = 0
    return image
    
def create_square(image, side_length, x, y):
        # Ensure the square fits within the image boundaries
    side_length = min(side_length, image.shape[0] - y, image.shape[1] - x)
    
     # Set the region defined by x, y, and side length to 1
    image[y:y + side_length, x:x + side_length] = 0.0
    
    return image

def prepare_data():

    # Data loading
    image_path = "C:/Users/mafrc/Desktop/Uni/PMC/CodePMC/final_dataset/"
    default_image_path = "C:/Users/mafrc/Desktop/Uni/PMC/CodePMC/trous/"

    images = []
    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            img = image.load_img(image_path+filename, target_size=(128, 128))
            images.append(image.img_to_array(img))
    images = np.array(images)
    print("images", images.shape)

    # Select 3 random images
    random_indices = np.random.choice(images.shape[0], nb_random_images-1, replace=False)
    random_images = images[random_indices]

    default_images = []  # List to store default images as NumPy arrays

    for filename in os.listdir(default_image_path):
        if filename.endswith(".jpg"):
            img = image.load_img(default_image_path+filename, target_size=(128, 128))
            default_images.append(np.array(image.img_to_array(img)))

    # Convert the list of default images to a NumPy array
    default_images = np.array(default_images)
    
    # Combine the two arrays
    random_images = np.concatenate((random_images, default_images), axis=0)

    # Preprocess the data
    random_images = random_images / 255.0

    # Delete the 'images' array
    del images

    return random_images

def show_inital_predict(random_images,  model):
    # Display the selected RGB images
    for i in range(nb_random_images):
        plt.subplot(2, nb_random_images, i + 1)
        plt.imshow(random_images[i])
        plt.title(f"Initial image {i}")
        plt.axis('off')

    # Assuming 'input_data' is your input data for prediction
    # Assuming your model expects input shapes like (height, width, channels)
    predictions = model.predict(random_images)

    for i in range(nb_random_images):
        plt.subplot(2, nb_random_images, i + 1 + nb_random_images)
        plt.imshow(predictions[i])
        plt.title(f"Predicted image {i}")
        plt.axis('off')

    plt.show()

import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def create_transformed_images_and_compare(original_image, square_size, image_numb, model, debug=False):
    # Initialize an array to store the transformed images
    transformed_images = []

    # Get the dimensions of the original image
    rows, cols, channels = original_image.shape

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
            current_image = np.copy(original_image)

            # Calculate the coordinates of the square
            square_top_left = (i * square_size, j * square_size)
            square_bottom_right = (square_top_left[0] + square_size, square_top_left[1] + square_size)

            # Black out the square for each channel
            for channel in range(channels):
                current_image[square_top_left[0]:square_bottom_right[0], square_top_left[1]:square_bottom_right[1], channel] = 0

            current_image_reshaped = current_image.reshape((-1, image_size, image_size, channels))
            
            # Generate image
            prediction = model.predict(current_image_reshaped)
            prediction = prediction.reshape(image_size, image_size, channels)

            # Test to compare
            prediction_unmodified = model.predict(original_image.reshape((-1, image_size, image_size, channels)))
            prediction_unmodified = prediction_unmodified.reshape(image_size, image_size, channels)

            # Compare the blacked out square for each channel separately
            ssim_values = []
            for channel in range(channels):
                #img1_channel = original_image[:,:,channel]
                img1_channel = prediction_unmodified[:,:,channel]
                img2_channel = prediction[:,:,channel]
                ssim_index, _ = ssim(img1_channel, img2_channel, full=True, data_range=1)
                ssim_values.append(ssim_index)

            # Average the SSIM values across channels
            avg_ssim = np.mean(ssim_values)
            print(avg_ssim)
            if avg_ssim < 0.95:

                if avg_ssim < worst_ssim:
                    worst_ssim = avg_ssim
                    worst_ssim_position = [i, j]
                    worst_ssim_square = current_image
                    worst_ssim_prediction = prediction

                if debug:
                    print(f"Flag in image {image_numb} at square {i} : {j} = {avg_ssim}")

                    # Create subplots for predicted images
                    plt.imshow(original_image)
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
                    plt.imshow(original_image)
                    plt.title(f"Predicted comparison at square {i} : {j}")
                    plt.axis('off')
                    plt.show()

    return worst_ssim, worst_ssim_position, worst_ssim_square, worst_ssim_prediction


if __name__ == '__main__':

    model = load_model()

    images = prepare_data()

    show_inital_predict(images, model)

    for i in range(nb_random_images):
        worst_ssim, worst_ssim_position, worst_ssim_square, worst_ssim_prediction = create_transformed_images_and_compare(images[i], square_size, i, model)

        if worst_ssim < 1:
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Plot the first RGB image
            axes[0].imshow(images[i])
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