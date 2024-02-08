"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/11/2024
Description: Contain the list of GAN model that will be used to train the GAN with. Each GAN model have a generator 
             and a discriminator. It is easy to change for one architecture or another thanks to this code. 
"""
import keras
from keras import layers
import tensorflow as tf

import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Activation
from keras.models import Model
from keras.preprocessing import image

class AeModels():
    def __init__(self, learning_rate: float=0.001):
        self.learning_rate = learning_rate

    def build_francois_chollet_autoencoder(self, input_shape: tuple=(784,), encoding_dim: int=32) -> keras.Model:
        """
        Francois Chollet auto-encoder (AE) architecture from this link:
        https://blog.keras.io/building-autoencoders-in-keras.html 
        Which is a Single fully-connected neural layer. 
        """
        #Define the input shape of the model.
        input_img = keras.Input(shape=input_shape)

        #Encoder
        Encoder = layers.Dense(encoding_dim, activation='relu')(input_img)
        #Decoder
        Decoder = layers.Dense(784, activation='sigmoid')(Encoder)

        #Define the input and output of the model
        model = keras.Model(inputs=input_img, outputs=Decoder)

        #Create the loss function 
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=opt, 
            loss='binary_crossentropy',
            metrics=['mean_absolute_error']
        )

        return model

    #TODO: Pass all parameters to tune
    def build_basic_cae(self, input_shape: tuple=(128, 128, 3)) -> keras.Model:
        """
        Model based on this kaggle notebook: https://www.kaggle.com/code/orion99/autoencoder-made-easy/notebook
        Tested using the subdivided dataset
        """
        input_layer = Input(shape=input_shape, name="INPUT")
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

        code_layer = MaxPooling2D((2, 2), name="CODE")(x)

        x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(code_layer)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)
        output_layer = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(x)
        output_layer = Activation('sigmoid')(output_layer)

         #Define the input and output of the model
        model = Model(input_layer, output_layer)

        #Create the loss function
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=opt, 
            loss='binary_crossentropy',
            metrics=['mean_absolute_error']
        )

        return model

