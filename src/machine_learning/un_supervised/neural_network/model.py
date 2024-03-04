#!/usr/bin/env python
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/11/2024
Description: Contain the list of GAN model that will be used to train the GAN with. Each GAN model have a generator 
             and a discriminator. It is easy to change for one architecture or another thanks to this code. 
"""

import keras
import tensorflow as tf

from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, LeakyReLU, Add, UpSampling2D, Activation, Dense, Input

from keras.models import Model

class AeModels():
    def __init__(self, learning_rate: float=0.001, monitor_loss: str='mean_absolute_error', monitor_metric: str='mean_squared_error'):
        self.learning_rate = learning_rate
        self.monitor_loss = monitor_loss
        self.monitor_metric = monitor_metric

    def build_francois_chollet_autoencoder(self, input_shape: tuple=(784,), encoding_dim: int=32) -> Model:
        """
        Francois Chollet auto-encoder (AE) architecture from this link:
        https://blog.keras.io/building-autoencoders-in-keras.html 
        Which is a Single fully-connected neural layer. 
        """
        #Define the input shape of the model.
        input_img = Input(shape=input_shape)

        #Encoder
        Encoder = Dense(encoding_dim, activation='relu')(input_img)
        #Decoder
        Decoder = Dense(784, activation='sigmoid')(Encoder)

        #Define the input and output of the model
        model = Model(inputs=input_img, outputs=Decoder)

        #Create the loss function 
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=opt, 
            loss=self.monitor_loss,
            metrics=[self.monitor_metric]
        )

        return model
    
    def convolutional_ae(self) -> Model:
        """
        Convolutional auto-encoder for denoising. architecture from this link:
        https://keras.io/examples/vision/autoencoder/ 
        """
        input = Input(shape=(256, 256, 3))

        #Encoder 
        x = Conv2D(32, (3,3), activation='relu', padding='same')(input)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        #Decoder
        x = Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)
        x = Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)
        x = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
        #Autoencoder
        model = keras.Model(inputs=input, outputs=x)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=opt, 
            loss=self.monitor_loss,
            metrics=[self.monitor_metric]
        )

        return model
    
    def build_basic_cae(self, input_shape: tuple=(256, 256, 3)) -> Model:
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
            loss=self.monitor_loss,
            metrics=[self.monitor_metric]
        )

        return model
    
    def aes_defect_detection(self) -> Model:
        """
        Convolution auto-encoder for defect detection found in the literature
        https://arxiv.org/pdf/2008.12977.pdf 
        """
        input_with_defect = Input(shape=(256, 256, 3))

        #Encoder
        e1 = Conv2D(16, (5,5), strides=2, padding='same')(input_with_defect)
        e1 = BatchNormalization()(e1)
        e1 = LeakyReLU(alpha=0.1)(e1)

        e2 = Conv2D(32, (5,5), strides=2, padding='same')(e1)
        e2 = BatchNormalization()(e2)
        e2 = LeakyReLU(alpha=0.1)(e2)

        e3 = Conv2D(64, (5,5), strides=2, padding='same')(e2)
        e3 = BatchNormalization()(e3)
        e3 = LeakyReLU(alpha=0.1)(e3)

        e4 = Conv2D(128, (5,5), strides=2, padding='same')(e3)
        e4 = BatchNormalization()(e4)
        e4 = LeakyReLU(alpha=0.1)(e4)

        e5 = Conv2D(256, (5,5), strides=2, padding='same')(e4)
        e5 = BatchNormalization()(e5)
        e5 = LeakyReLU(alpha=0.1)(e5)

        #Bottleneck layer 
        x = Conv2D(512, (5,5), strides=2, padding='same')(e5)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        #Decoder
        d5 = UpSampling2D()(x)
        d5 = Conv2D(256, (5,5), padding='same')(d5)
        d5 = BatchNormalization()(d5)
        d5 = Add()([d5, e5])
        d5 = LeakyReLU(alpha=0.1)(d5)

        d4 = UpSampling2D()(d5)
        d4 = Conv2D(128, (5,5), padding='same')(d4)
        d4 = BatchNormalization()(d4)
        d4 = Add()([d4, e4])
        d4 = LeakyReLU(alpha=0.1)(d4)

        d3 = UpSampling2D()(d4)
        d3 = Conv2D(64, (5,5), padding='same')(d3)
        d3 = BatchNormalization()(d3)
        d3 = Add()([d3, e3])
        d3 = LeakyReLU(alpha=0.1)(d3)

        d2 = UpSampling2D()(d3)
        d2 = Conv2D(32, (5,5), padding='same')(d2)
        d2 = BatchNormalization()(d2)
        d2 = Add()([d2, e2])
        d2 = LeakyReLU(alpha=0.1)(d2)
        
        d1 = UpSampling2D()(d2)
        d1 = Conv2D(16, (5,5), padding='same')(d1)
        d1 = BatchNormalization()(d1)
        d1 = Add()([d1, e1])
        d1 = LeakyReLU(alpha=0.1)(d1)

        output = UpSampling2D()(d1)
        output = Conv2D(3, (5,5), strides=1, padding='same')(output)
        output = Activation('sigmoid')(output)

        #Autoencoder
        model = Model(inputs=input_with_defect, outputs=output)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=opt, 
            loss=self.monitor_loss,
            metrics=[self.monitor_metric]
        )

        return model
