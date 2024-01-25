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

def autoencoder_fran_chol(learning_rate):
    """
    Francois Chollet auto-encoder (AE) architecture from this link:
    https://blog.keras.io/building-autoencoders-in-keras.html 
    This is the bench mark used because it is an easy model to start with 
    and was proposed by Francois Grondin. 
    INPUT
    :float Learning_rate: Size of the steps taken during the gradient descent
    OUTPUT
    :Objet model: Neural network architecture.
    """
    #Define the input shape of the model
    input_img = keras.Input(shape=(784,))

    #"Encoded" is the encoded representation of the input
    x = layers.Dense(32, activation='relu')(input_img)
    #"decoded" is the lossy reconstruction of the input
    x = layers.Dense(784, activation='sigmoid')(x)

    #Define the input and output of the model
    model = keras.Model(inputs=input_img, outputs=x)

    #Create the loss function 
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt, 
        loss='binary_crossentropy',
        metrics='mean_absolute_error'
    )

    return model