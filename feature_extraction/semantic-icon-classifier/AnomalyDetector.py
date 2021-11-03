#!/usr/bin/env python
from __future__ import print_function
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class AnomalyDetector(object):
    default_settings = {
        'x_train': 'data/x_train_class.npy',
        'y_train': 'data/y_train_embeddings.npy',
        'anomalies': 'data/anomalies_embeddings.npy',
        'test_valid': 'data/gmm_valid_class.npy',
        'test_invalid': 'data/gmm_invalid_class.npy',
    }

    def __init__(self, threshold=500, train=False):
        if train:
            self.x_train = np.load(self.default_settings['x_train'])
            self.y_train = np.load(self.default_settings['y_train'])
            self.y_train = np.zeros(len(self.x_train))
            self.model = GaussianMixture(n_components=128, verbose=True, covariance_type='full')
            self.inv_model = None
            self.anomalies = np.load(self.default_settings['anomalies'])
            self.y_test_proba = None
            self.inv_y_test_proba = None
            self.test_valid = np.load(self.default_settings['test_valid'])
            self.test_invalid = np.load(self.default_settings['test_invalid'])
        self.threshold = threshold
        self.inv_threshold = threshold + 100

    def invert(self):
        self.x_train, self.test_invalid = train_test_split(self.test_invalid)

    def train(self, filename='anomaly.pkl'):
        print(self.x_train.shape)
        self.model.fit(self.x_train)
        with open(filename, 'wb') as datagen_file:
            pickle.dump(self.model, datagen_file)

    def train_results(self):
        y_test_proba = self.model.score_samples(self.x_train)
        plt.plot(y_test_proba)
        plt.title('Predicted probabilities for train dataset')
        plt.savefig('bla_train.png')
        plt.clf()

    def load(self, filename='anomaly.pkl'):
        with open(filename, 'rb') as anomaly_file:
            self.model = pickle.load(anomaly_file)

    def load_inv(self, filename='inv_anomaly.pkl'):
        with open(filename, 'rb') as anomaly_file:
            self.inv_model = pickle.load(anomaly_file)

    def prep_tweak(self):
        self.y_test_proba = self.model.score_samples(np.vstack([self.test_valid, self.test_invalid]))
        if self.inv_model:
            self.inv_y_test_proba = self.inv_model.score_samples(np.vstack([self.test_valid, self.test_invalid]))

    def tweak(self, T=-500, T2=600, inverted=False):
        plt.plot(self.y_test_proba)
        plt.title('Predicted probabilities for test dataset')
        plt.savefig('bla.png')
        plt.clf()
        y_test_proba = np.zeros(self.y_test_proba.shape)
        y_test_proba[self.y_test_proba>=T] = int(inverted)
        y_test_proba[self.y_test_proba<T] = int(not inverted)
        if self.inv_model:
            y_test_proba[np.logical_and(self.y_test_proba < T2, self.inv_y_test_proba > self.y_test_proba)] = int(not inverted)
        y_actual = np.hstack([np.zeros(len(self.test_valid)), np.ones(len(self.test_invalid))])
        print('Classification report')
        print(classification_report(y_actual, y_test_proba))

    def predict(self, x):
        y_test_proba = self.model.score_samples(x)
        result = np.zeros(y_test_proba.shape)
        result[y_test_proba < self.threshold] = 1
        if self.inv_model:
            inv_y_test_proba = self.inv_model.score_samples(x)
            result[np.logical_and(y_test_proba < self.inv_threshold, inv_y_test_proba > y_test_proba)]
        return result


def main():
    ad = AnomalyDetector()
    ad.load()


if __name__ == '__main__':
    main()
