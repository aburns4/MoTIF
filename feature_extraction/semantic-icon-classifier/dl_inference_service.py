#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import argparse
import json
import pickle
import os

from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from PIL import Image

from cnn_pretrain import load_data, initialize_datagen
from AnomalyDetector import AnomalyDetector
import settings

class DlInferenceService(object):
    default_settings = {
        'model_path': './saved_models/small_cnn_weights_50_512.h5',
        'validation_metadata': './saved_models/validation_metadata.json',
        'datagen': './saved_models/datagen.pkl',
        'anomaly_model': './saved_models/anomaly.pkl',
        'inv_anomaly_model': './saved_models/inv_anomaly.pkl',
    }

    def __init__(self, model_path=None, class_names=None, datagen=None, anomaly=True, input_size=32):
        if not model_path:
            model_path = self.default_settings['model_path']

        if not class_names:
            with open(self.default_settings['validation_metadata']) as class_names_file:
                class_names = json.load(class_names_file)['class_names']

        if not datagen:
            with open(self.default_settings['datagen'], 'rb') as datagen_file:
                datagen = pickle.load(datagen_file)

        self.anomaly_model = None
        if anomaly:
            self.anomaly_model = AnomalyDetector()
            self.anomaly_model.load(self.default_settings['anomaly_model'])
            self.anomaly_model.load_inv(self.default_settings['inv_anomaly_model'])

        self.model = load_model(model_path)
        self.class_names = class_names
        self.datagen = datagen
        self.input_size = input_size
        self.embeddings_model = None
        
    def _load_image(self, image_path):
        image = self._preprocess_image(Image.open(image_path))
        return self._image_to_numpy_array(image)

    def _preprocess_image(self, image):
        size = (self.input_size, self.input_size)
        return image.convert('L').resize(size, Image.ANTIALIAS)

    def _image_to_numpy_array(self, image):
        size = (self.input_size, self.input_size, 1)
        output = np.asarray(image, dtype='int32')
        output = output.astype('float32')
        output /= 255
        return output.reshape(size)

    def _image_paths_to_numpy_array(self, image_paths):
        image_count = len(image_paths)
        x_shape = (image_count, self.input_size, self.input_size, 1)
        x = np.zeros(x_shape)

        for idx, image_path in enumerate(image_paths):
            x[idx] = self._load_image(image_path)

        return x

    def _initialize_datagen(self):
        datagen = ImageDataGenerator(
            zca_whitening=True,
        )
        return datagen

    def _predict(self, x):
        x = self.datagen.flow(x, batch_size=len(x), shuffle=False).next()
        result = self.model.predict(x)
        return result

    def _predictions_to_class(self, result):
        predicted_class = np.argmax(result, axis=1)
        return predicted_class

    def _initialize_embedding_model(self, embedding_layer_name='embedding'):
        self.embedding_model = Sequential()
        for layer in self.model.layers:
            self.embedding_model.add(layer)
            if layer.name == embedding_layer_name:
                break

    def identify_anomalies(self, x):
        pass

    def generate_embeddings(self, x):
        if not self.embeddings_model:
            self._initialize_embedding_model()

        x = self.datagen.flow(x, batch_size=len(x), shuffle=False).next()
        embeddings = self.embedding_model.predict(x)
        return embeddings

    def classify(self, x):
        predictions = self._predict(x)
        print(predictions.shape)
        anomalies = np.zeros(len(predictions))
        if self.anomaly_model:
            anomalies = self.anomaly_model.predict(predictions)
        return [self.class_names[prediction] if not anomalies[idx] else 'anomaly'
                for idx, prediction in enumerate(self._predictions_to_class(predictions))]

    def classify_paths(self, image_paths):
        x = self._image_paths_to_numpy_array(image_paths)
        return self.classify(x)

    def classify_images(self, images):
        image_count = len(images)
        x_shape = (image_count, self.input_size, self.input_size, 1)
        x = np.zeros(x_shape)

        for idx, image in enumerate(images):
            x[idx] = self._image_to_numpy_array(self._preprocess_image(image))

        return self.classify(x)

    def create_npy(self, image_paths, out_path):
        x = self._image_paths_to_numpy_array(image_paths)
        np.save(out_path, x)

    def make_preprocessed_images(self, npy, augment=False):
        x = np.load(npy)
        if augment:
            x = self.datagen.flow(x, batch_size=10, shuffle=False).next()
        for idx, row in enumerate(x):
            print(row.shape)
            bla = Image.fromarray(row.reshape(32, 32) * 255).convert('RGB')
            out_dir = '{}_preprocessed{}'.format(npy, '_augmented' if augment else '')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            bla.save('{}/{}.png'.format(out_dir, idx))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help=('Path to the h5 model file'),
                        default='./sm_fancy/small_cnn_weights_50_512.h5')
    parser.add_argument('--class_names',
                        help=('Path to json with a class_names list'),
                        default='./data/validation_metadata.json')
    parser.add_argument('--datagen',
                        help='Path to pickled datagen',
                        default='./sm_fancy/datagen.pkl')
    parser.add_argument('--gen_npy',
                        help='Where do you wanna put the npy?')
    parser.add_argument('--preprocess',
                        help='Preprocess a npy file.')
    parser.add_argument('--augment',
                        help='Preprocess and augment a npy file')
    parser.add_argument('image_paths', nargs='*')
    args = parser.parse_args()

    inference_service = DlInferenceService()

    if args.gen_npy and args.image_paths:
        inference_service.create_npy(args.image_paths, args.gen_npy)
        return

    if args.preprocess or args.augment:
        inference_service.make_preprocessed_images(args.preprocess or args.augment, args.augment)
        return

    if args.image_paths:
        predictions = inference_service.classify_paths(args.image_paths)
        print(predictions)


if __name__ == '__main__':
    main()
