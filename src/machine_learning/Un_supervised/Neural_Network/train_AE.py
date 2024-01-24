import keras
from keras import layers
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

#This is the size of our encoded representations
encoding_dim = 32 # Float 32 -> compresseion of factor 24.5, assuming the input is 784 floats. (Flatten 28x28 pixel grayscale images)

#This is our input image
input_img = keras.Input(shape=(784,))
#"Encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
#"decoded" is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)

#This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

#This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

#This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
#Retrive the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
#Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

#Create the loss function 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#Prepare the data and import it from the MNIST
(x_train, _), (x_test, _) = mnist.load_data()

#Normalize all value between 0 and 1. 
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

#Know let's train our model over 50 epochs
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

#Encode and decode some digits
#Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#Plot the image
n = 10 #How many digits we will display
plt.figure(figsize=(20,4))
for i in range(n):
    #Display original 
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #Display the reconstruction 
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
