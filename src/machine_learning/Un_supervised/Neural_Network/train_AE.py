from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

import callbacks as cb
#import hyper_parameters_tuner as hp_tuner
import model as mod


epochs = 5
batch_size = 256

test_name = 'First'
filepath_weights = f'/home/jean-sebastien/Documents/s7/PMC/results_ae/best_model_weights/{test_name}/'
monitor_metric = 'mean_absolute_error'
mode_metric = 'min'
verbose = 1


def normalize(input_train, input_test):
    input_train = input_train.astype('float32') / 255.
    input_test = input_test.astype('float32') / 255.
    input_train = input_train.reshape((len(input_train), np.prod(input_train.shape[1:])))
    input_test = input_test.reshape((len(input_test), np.prod(input_test.shape[1:])))

    return input_train, input_test

def de_normalize(input_test, result_test):
    original_image_height = 28
    original_image_width = 28

    input_test = input_test.reshape((len(input_test), original_image_height, original_image_width))
    result_test = result_test.reshape((len(result_test), original_image_height, original_image_width))
    input_test = input_test * 255
    result_test = result_test * 255

    return input_test, result_test

def train(model, input_train, input_test, epochs, batch_size, callbacks):
    history = model.fit(
        x=input_train, 
        y=input_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        validation_data=(input_test, input_test),
        shuffle=True
    )

    return history

def evaluate_model(model, input_test, batch_size=32, verbose=1):
    test_loss = model.evaluate(
        x=input_test,
        y=input_test,
        batch_size=batch_size,
        verbose=verbose
    )
    print(f'Test loss: {test_loss:.3f}')

    return test_loss

def plot_graph(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'blue', label='Training_loss')
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
    return model.predict(
        x=input_test,
        batch_size=None,
        verbose=1,
        steps=None,
        callbacks=None, 
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )

def createPredImg(input_train, input_test):
    # Redimensionner les images de 784 à 28x28
    input_train_reshaped = np.reshape(input_train, (28, 28))
    input_test_reshaped = np.reshape(input_test, (28, 28))
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Input data vs. Results')

    axes[0].set_title('Inputs')
    axes[0].imshow(input_train_reshaped, cmap='gray')
    axes[1].set_title('Results')
    axes[1].imshow(input_test_reshaped, cmap='gray')

    plt.show()


if __name__ == '__main__':
    #Data loading
    (input_train, _), (input_test, _) = mnist.load_data()

    #Normalizing the data (0-1)
    input_train_norm, input_test_norm = normalize(input_train, input_test)

    #Building model 
    model = mod.AeModels(learning_rate=0.001)
    build_model = model.build_francois_chollet_autoencoder(input_shape=(784,), encoding_dim=32)

    #Training model
    callback = cb.TrainingCallbacks(filepath=filepath_weights, monitor=monitor_metric, mode=mode_metric, verbose=verbose)
    callbacks_list = callback.get_callbacks()

    history = train(build_model, input_train_norm, input_test_norm, epochs, batch_size, callbacks_list)

    #Plot training history
    plot_graph(history)

    #Test the model
    results_norm = prediction(build_model, input_test_norm)

    #De-normalize Data
    de_normalize(input_test_norm, results_norm)

    #Plot result
    for i in range(5):
        createPredImg(input_test[i], results_norm[i])
