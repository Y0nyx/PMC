#!/user/bin/env python

"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/11/2024
Description: Training the Neural Network, which is, in this case a GAN model. A GAN model is composed of a Generator
             and a Discriminator. The role of the generator is to generate images from random numbers and will try to 
             reproduce the images of interest. The discriminator will have to find in the image it receive is a real 
             image or an image that have been reproduced by the GAN. The Generator and Discriminator are playing a 
             game and when the generator win, the weights of the discriminator get updated for the next itteration and
             when the discriminator win, the weights of the generator get updated for the next itteration. 
             At the moment we are using GAN from Francois Chollet book: Deep learning with python p.401

Note:        In our case the generator will learn to reproduce images that look like the input data. For it part, the 
             discriminator will learn to know if the image is a real image or a fake one. Once both network became really
             good at what they do, the discriminator will be used to know if there is a welding error in a piece. Indeed, 
             it will know if there is an error if the input image do not look like the regenerate image of the generator. 

Next itteration: Try to use smaller images as the input. It could be possible to divise the original image in smaller pieces
                 like Mathias find in the litterature.

                 - At the moment everything will be here. But it will be important to use other's folders in the next itteration
"""

#####################################################################################################################
#                                                      Imports                                                      # 
#####################################################################################################################
from tensorflow import keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf



#####################################################################################################################
#                                                     Variables                                                     #
#####################################################################################################################
SUMMARY = True #Flag to show the structure used to build the architecture


#####################################################################################################################
#                                                     Functions                                                     #
#####################################################################################################################



######################################################################################################################
#                                                        Main                                                        #
######################################################################################################################
if __name__ == '__main__':
    #Data processing
    #Creating a dataset from a directory of images
    dataset = keras.utils.image_dataset_from_directory(
        directory="/home/jean-sebastien/Documents/s7/PMC/Data/img_align_celeba", 
        label_mode=None,
        image_size=(64,64),
        batch_size=32,
        smart_resize=True
    )
    #Rescalling the images
    dataset = dataset.map(lambda x: x / 255.)

    #Displaying the first image
    for x in dataset:
        plt.axis("off")
        plt.imshow((x.numpy() * 255).astype("int32")[0])
        plt.show()
        break

    
    #Network architecture
    #Discriminator
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 3)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2), 
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

    if SUMMARY:
        discriminator.summary()

    #Generator
    latent_dim = 128
    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim)),
            layers.Dense(8 * 8 * 128), 
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

    if SUMMARY:
        generator.summary()

    #The GANs model 
    class GAN(keras.Model):
        def __init__(self, discriminator, generator, latent_dim):
            super().__init__()
            self.discriminator = discriminator
            self.generator = generator
            self.latent_dim = latent_dim
            #Sets up metrics to track the two losses over each training epoch
            self.d_loss_metric = keras.metrics.Mean(name="d_loss")
            self.g_loss_metric = keras.metrics.Mean(name="g_loss")

        def compile(self, d_optimizer, g_optimizer, loss_fn):
            super(GAN, self).compile()
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer
            self.loss_fn = loss_fn
            
        @property
        def metrics(self):
            return [self.d_loss_metric, self.g_loss_metric]
        
        def train_step(self, real_images):
            #Sample random points in the latent space
            batch_size = tf.shape(real_images)[0]
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim))
            #Decodes them to fake images
            generated_images = self.generator(random_latent_vectors)
            #Combines them with real images
            combined_images = tf.concat([generated_images, real_images], axis=0)
            #Assembles labals, discriminating real from fake images
            labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
                axis=0
            )

            #Add random noise to the labels
            labels += 0.5 * tf.random.uniform(tf.shape(labels))

            #Train the discriminator
            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_images)
                d_loss = self.loss_fn(labels, predictions)

            #Samples random points in the latent space
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim))
            
            #Assembles labels that say 'these are all real images' it's a lie!
            misleading_labels = tf.zeros((batch_size, 1))

            #Train the generator
            with tf.GradientTape() as tape:
                predictions = self.discriminator(
                    self.generator(random_latent_vectors))
                g_loss = self.loss_fn(misleading_labels, predictions)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(
                zip(grads, self.generator.trainable_weights))
            
            self.d_loss_metric.update_state(d_loss)
            self.g_loss_metric.update_state(g_loss)
            return {"d_loss": self.d_loss_metric.result(),
                    "g_loss": self.g_loss_metric.result()}
        
    #CALLBACK: samples generated images during training
    class GANMonitor(keras.callbacks.Callback):
        def __init__(self, num_img=3, latent_dim=128):
            self.num_img = num_img
            self.latent_dim = latent_dim

        def on_epoch_end(self, epoch, logs=None):
            random_latent_vectors = tf.random.normal(
                shape=(self.num_img, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)
            generated_images *= 255
            generated_images.numpy()

            for i in range(self.num_img):
                img = keras.utils.array_to_img(generated_images[i])
                img.save(f"generated_img_{epoch:03d}_{i}.png")

    #TRAINING: Compiling and training the GAN
    #Interesting results are starting to be seen after epoch 20
    epochs = 100

    gan = GAN(discriminator=discriminator, generator=generator,
              latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    gan.fit(
        dataset, epochs=epochs,
        callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
    )

            