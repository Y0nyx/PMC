import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import time

SUMMARY = True
start = time.time()
#__________________ VAE encoder network __________________ 
#Dimensionality of the latent space: a 2D plane
latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32,3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
#The input image ends up being encoded into these two parameters
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

if SUMMARY:
    encoder.summary()

#__________________ Latent-space-sampling layer __________________ 
class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        #Draw a batch of random vectors
        epsilon = tf.random.normal(shape=(batch_size, z_size))

        #Apply thge VAE sampling formula
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#______________________ VAE decoder network ______________________ 
#Mapping latent space points to images
#Input whre will be feed z
latent_inputs = keras.Input(shape=(latent_dim))
#Produce the same number of coefficients that we had at the level of the Flatten layer in the encoder
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
#Revert the Flatten layer of the encoder.
x = layers.Reshape((7, 7, 64))(x)
#Revert the Conv2D layers of the encoder
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
#The output ends up with shape (28, 28, 1).
decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

if SUMMARY:
    decoder.summary()

#______________ VAE model with custom train_step() ______________ 
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        #We use these metrics to keep track of the loss average over each epoch.
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property 
    #We list the metrics in the metrics property to enable the model to reset them after
    #each epoch (or between multiple calls to fit()/evaluate()).
    def metrics(self):
        return [self.total_loss_tracker, 
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]
    def train_step(self, data):
        with tf.GradientTape() as tape:
            #We sum the reconstruction loss over the spatial dimensions (axes 1 and 2)
            #and take its mean over the batch dimension. 
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            #Add the regularization term (Kullback-Leibler divergence)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - 
                                tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "total_loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
            
#__________________ Training the VAE __________________ 
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
#We train all the MNIST digits, so we concatenate the training and test samples. 
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
#Note that we dont pass argument in compile(), since the loss is already part of the train_step().
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
#Note that we dont pass targets in fit(), since train_step() doesnt expect any. 
vae.fit(mnist_digits, epochs=20, batch_size=32)

end = time.time()
print(f"The code took {end-start} to execute.")
#__ Sampling a grid of images from the 2D latent space __ 
#We will display a grid of 30 x 30 digits(900 digits total).
n = 30
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

#Sample points linearly on a 2D grid. 
grid_x = np.linspace(-1, 1, n)
grid_y = np.linspace(-1, 1, n)[::-1]

#Iterate over grid locations. 
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        #For each location, sample a digit and add it to our figure. 
        z_sample = np.array([[xi, yi]])
        x_decoded = vae.decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[
            i * digit_size : (i + 1) * digit_size, 
            j * digit_size : (j + 1) * digit_size,
        ] = digit
plt.figure(figsize=(15, 15))
start_range = digit_size // 2
end_range = n * digit_size + start_range
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.axis("off")
plt.imshow(figure, cmap="Greys_r")
plt.show()
