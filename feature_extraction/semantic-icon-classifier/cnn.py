import os
import numpy as np
import random

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.core import Reshape

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(
        K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


class Cnn:
    """The CNN part of the pipeline"""
    def __init__(self, model_file, weights_file, cnn_params):
        random.seed()
        self.model_history = None
        self.batch_size = cnn_params["batch_size"]
        self.embedding_size = cnn_params["embedding_size"]
        self.save_dir = cnn_params["save_dir"]

        with open(model_file, 'r') as json_file:
            loaded_model_json = json_file.read()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(weights_file)

        # we remove the top layer (the classifier) to get the part of the model that
        # embeds icons into a lower dimensional space
        self.embedding_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(index=-2).output)

        # we create a new network that forces images to map closer to the
        # centroids they are mapped to
        input_a = Input(shape=(32, 32, 1))  # we feed in the image here
        input_b = Input(shape=(32,))  # we feed in the centroids here

        processed_a = self.embedding_model(input_a)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, input_b])
        self.main_model = Model([input_a, input_b], distance)

    def get_embeddings(self, inputs):
        return self.embedding_model.predict(inputs)

    def train(self, points, centroids, epochs, lr=0.00001):
        opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)
        self.main_model.compile(loss=contrastive_loss, optimizer=opt)

        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,
            # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,
            # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            preprocessing_function=lambda t: random.choice([t, 1 - t]), # randomly invert images
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization

        datagen.fit(points)

        def generator():
            while True:
                my_flow = datagen.flow(
                    points, centroids, batch_size=self.batch_size)
                for i in range(points.shape[0] // self.batch_size):
                    (augmented_points, their_centroids) = next(my_flow)
                    yield ([augmented_points, their_centroids], np.ones((self.batch_size)))

        my_generator = generator()
        x, y = next(my_generator)

        self.model_info = self.main_model.fit_generator(generator(),
                                                        steps_per_epoch=points.shape[0] // self.batch_size,
                                                        epochs=epochs,
                                                        workers=1)

    def _plot_model_history(self, model_history, filename):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # summarize history for accuracy
        axs[0].plot(range(1, len(model_history.history['acc']) + 1),
                    model_history.history['acc'])
        axs[0].plot(range(1,
                          len(model_history.history['val_acc']) + 1),
                    model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,
                                    len(model_history.history['acc']) + 1),
                          len(model_history.history['acc']) / 10)
        axs[0].legend(['train', 'val'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1, len(model_history.history['loss']) + 1),
                    model_history.history['loss'])
        axs[1].plot(range(1,
                          len(model_history.history['val_loss']) + 1),
                    model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,
                                    len(model_history.history['loss']) + 1),
                          len(model_history.history['loss']) / 10)
        axs[1].legend(['train', 'val'], loc='best')
        fig.savefig(filename)
        plt.close(fig)

    def save_weights(self, filename):
        self.model.save_weights(filename, overwrite=True)
