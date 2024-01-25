import keras
from keras import layers
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

from model import *

epochs = 50
batch_size = 256

(input_train, _), (input_test, _) = mnist.load_data()

def normalize(input_train, input_test):
    input_train = input_train.astype('float32') / 255.
    input_test = input_test.astype('float32') / 255.
    input_train = input_train.reshape((len(input_train), np.prod(input_train.shape[1:])))
    input_test = input_test.reshape((len(input_test), np.prod(input_test.shape[1:])))

    return input_train, input_test

def train(model, input_train, input_test, epochs):
    history = model.fit(
        x=input_train, 
        y=input_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=None,
        verbose=1,
        validation_data=(input_test, input_test),
        shuffle=True
    )

    return history

def evaluate_model(model, input_test):
    test_loss = model.evaluate(
        x=input_test,
        y=input_test,
        batch_size=1,
        verbose=1
    )
    print(f'Test loss: {test_loss:.3f}')

def plot_graph(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'blue', lable='Training_loss')
    plt.plot(epochs, val_loss, 'orange', label='validation_loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['mean_absolute_error']
    val_acc = history.history['val_mean_absolute_error']
    plt.plot(epochs, acc, 'blue', label='Training accuracy')
    plt.plot(epochs, val_acc, 'orange', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def prediction(model, input_test):
    predictions = model.predict(
        x=input_test,
        batch_size=None,
        verbose=1,
        steps=None,
        callbacks=None, 
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )

    return predictions

def createPredImg(input_train, input_test):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Input data vs. Results')

    axes[0].set_title('Inputs')
    axes[0].imshow(input_train, origin='lower')
    axes[1].set_title('Results')
    axes[1].imshow(input_test, origin='lower')

    plt.show()