"""
Author(s): Mathias Gagnon, Jean-Sebastien Giroux
Contributor(s): 
Date: 01/25/2024
Description: Training and Testing(*temporaty*) of the neural network.
"""
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from keras.preprocessing import image as keras_image
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback
import model as mod
from Unsupervised_data_pipeline.data_pipeline import data_pipeline
import cv2


#TODO: Faire un objet image pour standardiser notation
#======================
#Variables globales d'image
#======================
image_height = 128
original_image_width = 128
channels = 3
normalization_factor = 255

#TODO: Faire un objet
#======================
#Variables globales de dataset
#======================
dataset_path = "../../../../../Datasets/grosse_piece_test"
input_train_norm = []
input_test_norm = []
input_train_aug_norm = []
input_test_aug_norm = []

nb_random_images = 3

#TODO: Mettre de vrais hyperparametres ici
sweep_config = {
  'method': 'random', 
  'metric': {
      'name': 'val_loss',
      'goal': 'minimize'
  },
  'early_terminate':{
      'type': 'hyperband',
      'min_iter': 5
  },
  'parameters': {
      'batch_size': {
          'values': [32]
      },
      'learning_rate':{
          'values': [0.001]
      },
      'epochs':{
        'values':[30]
      }
  }
}

#TODO: Make a single helper class with normalize and other helper methods
def normalize(train, test):
    train = train.astype('float32') / normalization_factor
    test = test.astype('float32') / normalization_factor

    return train, test

#TODO: On ne devrait pas reshape dans la denormalization, ce n'est pas la bonne place 
def de_normalize(model_input, model_output):
    model_input = model_input.reshape((len(model_input), image_height, original_image_width, channels))
    model_output = model_output.reshape((len(model_output), image_height, original_image_width, channels))

    model_input = model_input * normalization_factor
    model_output = model_output * normalization_factor

    return model_input, model_output

#TODO: Mieux nommer mes trucs
def train(model, input_train_norm, input_train_aug_norm, input_test_norm, input_test_aug_norm, epochs, batch_size, callbacks):
    print("=====================================")
    print("Fit with train dataset in arrays of shape: ", input_train_norm.shape, " | ", input_train_aug_norm.shape)
    print("=====================================")

    wandb_save_image_sample(input_train_norm[np.random.randint(0, 1)], "Train expected output")
    wandb_save_image_sample(input_train_aug_norm[np.random.randint(0, 1)], "Train input (black square image)")

    print("=====================================")
    print("Fit with test dataset in arrays of shape: ", input_test_aug_norm.shape, " | ", input_test_norm.shape)
    print("=====================================")

    wandb_save_image_sample(input_train_norm[np.random.randint(0,1)], "Test expected output")
    wandb_save_image_sample(input_train_aug_norm[np.random.randint(0,1)], "Test input (black square image)")

    model.fit(
        x=input_train_aug_norm, 
        y=input_train_norm,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        validation_data=(input_test_aug_norm, input_test_norm),
        shuffle=True
    )

def wandb_save_image_sample(image, name):

    plt.imshow(image)
    plt.title(name)
    plt.axis('off')

    # Log the plot
    wandb.log({name: plt})

def apply_random_blackout(images, blackout_size=(32, 32)):
    augmented_images = images.copy()

    for i in range(images.shape[0]):
        # Randomly select the position to blackout
        x = np.random.randint(0, images.shape[1] - blackout_size[0] + 1)
        y = np.random.randint(0, images.shape[2] - blackout_size[1] + 1)

        # Black out the selected region for each channel
        for channel in range(channels):
            augmented_images[i, x:x+blackout_size[0], y:y+blackout_size[1], channel] = 0.0

    return augmented_images

def load_data():

    images = []

    print("loading images")
    #Data loading
    for filename in os.listdir(dataset_path):
        if filename.endswith(".png"):
            img = keras_image.load_img(os.path.join(dataset_path, filename))
            images.append(keras_image.img_to_array(img))
    images = np.array(images)

    print("init pipeline")
    data_pipeline_obj = data_pipeline(image_height, original_image_width)
    sub_image_list = []

    for index, image in enumerate(images):
        print("Subdividing images")
        data_pipeline_obj.set_image(image)
        sub_images = data_pipeline_obj.subdivise() # This returns a list of sub-images
        sub_image_list.append(sub_images)

        if index < 2:
            # Save the first two sub-images for the first two images
            for sub_index, sub_image in enumerate(sub_images[:10]): # Adjust the slice as needed
            # Convert sub-image from BGR to RGB (OpenCV uses BGR by default)
                sub_image_rgb = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)
            # Save the sub-image
                cv2.imwrite(os.path.join("validation_folder/", f"image_{index}_sub_{sub_index}.png"), sub_image_rgb)

    images = np.array(sub_image_list)
    images = images.reshape(-1, 128, 128, 3)

    print("=====================================")
    print("Loaded image np.array of shape: ", images.shape)
    print("=====================================")

    # Split the dataset into training and testing sets (70/30 split)
    input_train, input_test = train_test_split(images, train_size=0.7, test_size=0.3, random_state=42)
    print("=====================================")
    print("Splitted dataset in arrays of shape: ", input_train.shape, " | ", input_test.shape)
    print("=====================================")

    del images

    train_augmented = apply_random_blackout(input_train)
    test_augmented = apply_random_blackout(input_test)
    print("=====================================")
    print("Augmented splitted dataset in arrays of shape: ", train_augmented.shape, " | ", test_augmented.shape)
    print("=====================================")


    #Normalizing the data (0-1)
    input_train_norm, input_test_norm = normalize(input_train, input_test)
    input_train_aug_norm, input_test_aug_norm = normalize(train_augmented, test_augmented)

    return input_train_norm, input_train_aug_norm, input_test_norm, input_test_aug_norm

def build_and_train_model():
    wandb.init(project="local_CAE_training_v0")
    #Building model 
    model_factory = mod.AeModels(learning_rate=wandb.config.learning_rate)
    model = model_factory.build_basic_cae()
    #Training model
    callbacks = []

    #Get data
    input_train_norm, input_train_aug_norm, input_test_norm, input_test_aug_norm = load_data()
    train(model, input_train_norm, input_train_aug_norm, input_test_norm, input_test_aug_norm, wandb.config.epochs, wandb.config.batch_size, callbacks)

    predict_sample_and_save(model, input_test_norm[0], input_test_aug_norm)


def predict_sample_and_save(model, input_test_norm, input_test_aug_norm):
    # Reshape input data to include a singleton batch dimension
    sample_to_predict = input_test_aug_norm[0]

    # Reshape the selected sample to have the shape (1, 128, 128, 3)
    input_test_aug_norm_single = np.expand_dims(sample_to_predict, axis=0)

    # Predict using the model
    prediction = model.predict(input_test_aug_norm_single)

    # Save images
    wandb_save_image_sample(input_test_norm, "Prediction expected output")
    wandb_save_image_sample(sample_to_predict, "Prediction input (black square image)")
    wandb_save_image_sample(prediction[0], "Prediction input")


if __name__ == '__main__':

    sweep_id = wandb.sweep(sweep=sweep_config, project="local_CAE_training_v0.2")

    # Run the sweep
    wandb.agent(sweep_id, function=build_and_train_model)
