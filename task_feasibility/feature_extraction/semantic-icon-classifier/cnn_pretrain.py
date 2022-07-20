#!/usr/bin/env python
"""Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
"""
from __future__ import print_function
import os
import json
import time
import random
import argparse
import pickle
import glob
from PIL import Image

import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, merge, Concatenate, concatenate
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras import backend as K
from keras.utils import plot_model
from sklearn import metrics

from cnn import eucl_dist_output_shape, contrastive_loss, euclidean_distance
import settings

def euc_dist(x):
    'Merge function: euclidean_distance(u,v)'
    s = x[0] - x[1]
    output = (s ** 2).sum(axis=1)
    output = K.reshape(output, (output.shape[0],1))
    return output

def euc_dist_shape(input_shape):
    'Merge output shape'
    shape = list(input_shape)
    outshape = (shape[0][0],1)
    return tuple(outshape)

def accuracy(x, y, model, class_names, cm_filename, datagen, words=None, anomaly=False):
    num_classes = len(class_names)
    print(len(set(np.argmax(y, axis=1))))
    if words is not None:
        x, y = word_datagen(datagen, x, words, y, len(x)).next()
    else:
        x, y = datagen.flow(x, y, batch_size=len(x), shuffle=False).next()
    result = model.predict(x)

    anomaly_class = num_classes
    anomalies = np.zeros(len(result))
    if anomaly:
        from dl_inference_service import DlInferenceService
        dlis = DlInferenceService()
        anomalies = dlis.anomaly_model.predict(result)
        class_names += ['anomaly']
        num_classes += 1
    predicted_class = np.argmax(result, axis=1)
    predicted_class[anomalies == 1] = anomaly_class
    true_class = np.argmax(y, axis=1)
    if anomaly:
        predicted_class[0] = anomaly_class
        true_class[0] = anomaly_class
    print(len(set(true_class)))
    print(predicted_class[0:10])
    print(true_class[0:10])
    np.save('small.npy', x[0:10])
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct) / result.shape[0]
    return accuracy * 100, predicted_class, true_class, x


def load_data(data_folder):
    x_train = np.load(data_folder("training_x.npy"))
    y_train = np.load(data_folder("training_y.npy"))
    x_test = np.load(data_folder("validation_x.npy"))
    y_test = np.load(data_folder("validation_y.npy"))

    # Print Training and Testing data dimension
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Pre-process data, so value is between 0.0 and 1.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    num_classes = np.unique(y_train).shape[0]
    # Print Unique Icon Classes, 99 classes
    # print(np.unique(y_train))
    # print(num_classes, ' classes')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, num_classes

def load_word_data(data_folder):
    x_train = np.load(data_folder("training_x_words.npy"))
    x_test = np.load(data_folder("validation_x_words.npy"))

    return x_train, x_test


def create_model(embedding_size, num_classes, model_type='', siamese=False, conv_only=False):
    model = Sequential()

    if model_type == 'simple':
        model.add(Conv2D(32, (3, 3), padding='same', activation='elu', input_shape=(32, 32, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(fmp)
        model.add(Dropout(0.15))

        model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5, name='dropout'))

        model.add(Flatten(name='flattened'))
        if conv_only:
            return model

        model.add(Dense(embedding_size, activation='elu', name='embedding'))
        if siamese:
            return model

        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return model

    model.add(Conv2D(384, (3, 3), padding='same', activation='elu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(384, (1, 1), padding='same', activation='elu'))
    model.add(Conv2D(384, (2, 2), padding='same', activation='elu'))
    model.add(Conv2D(640, (2, 2), padding='same', activation='elu'))
    model.add(Conv2D(640, (2, 2), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(640, (1, 1), padding='same', activation='elu'))
    model.add(Conv2D(768, (2, 2), padding='same', activation='elu'))
    model.add(Conv2D(768, (2, 2), padding='same', activation='elu'))
    model.add(Conv2D(768, (2, 2), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(768, (1, 1), padding='same', activation='elu'))
    model.add(Conv2D(896, (2, 2), padding='same', activation='elu'))
    model.add(Conv2D(896, (2, 2), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(896, (3, 3), padding='same', activation='elu'))
    model.add(Conv2D(1024, (2, 2), padding='same', activation='elu'))
    model.add(Conv2D(1024, (2, 2), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(1024, (1, 1), padding='same', activation='elu'))
    model.add(Conv2D(1152, (2, 2), padding='same', activation='elu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    if conv_only:
        return model

    model.add(Dense(embedding_size, activation='elu', name='embedding'))

    model.add(Dense(num_classes, activation='softmax', name='classification'))
    return model


def initialize_model(embedding_size, num_classes, model_type=''):
    p_ratio = [1.0, 1.44, 1.73, 1.0]
    fmp = Lambda(lambda x: tf.nn.fractional_max_pool(x, p_ratio)[0])

    model = create_model(embedding_size, num_classes, model_type=model_type)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-4)

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model


def initialize_word_model(embedding_size, num_classes, model_type=''):
    base_model = create_model(embedding_size, num_classes, model_type=model_type, conv_only=True)
    left_input = Input((32, 32, 1), name='left')
    right_input = Input((835,), name='right')

    word_model = Sequential()
    word_model.add(Dense(1024, activation='elu', input_shape=(835,)))
    word_model.add(Dropout(0.5))
    word_model.add(Dense(512, activation='elu'))
    word_model.add(Dropout(0.5))

    encoded_l = base_model(left_input)
    encoded_r = word_model(right_input)

    both = concatenate([encoded_l, encoded_r])
    out = Dense(4096, activation='elu')(both)

    sigh = Dense(4096, activation='elu')(out)
    sigh = Dropout(0.5)(sigh)
    sigh = Dense(embedding_size, activation='elu')(sigh)
    sigh = Dropout(0.5)(sigh)
    sigh = Dense(num_classes, activation='softmax')(sigh)

    model = Model(input=[left_input, right_input], output=sigh)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-4)

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model


def initialize_siamese_model(embedding_size, num_classes, model_type=''):
    in_dim = (32, 32, 1)
    left_input = Input((32, 32, 1), name='left')
    right_input = Input((32, 32, 1), name='right')

    base_model = create_model(embedding_size, num_classes, model_type=model_type, siamese=True)
    file_model = keras.models.load_model('/home/ranjitha/code/mobile-embeddings/clustering_with_cnns/saved_models_f_simple/small_cnn_weights_10_512.h5')
    print("file layers")
    print(file_model.layers)
    base_model = Sequential()
    for layer in file_model.layers[:-1]:
        layer.training = False
        base_model.add(layer)
    base_model.add(Dense(4096, activation='elu'))
    base_model.add(Dropout(0.2))
    base_model.add(Dense(embedding_size, activation='elu'))
    base_model.add(Dense(num_classes, activation='softmax'))

    encoded_l = base_model(left_input)
    encoded_r = base_model(right_input)
    print(type(encoded_l))
    opt = keras.optimizers.rmsprop()
    both = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])

    model = Model(input=[left_input, right_input], output=both)

    # train
    model.compile(loss=contrastive_loss, optimizer=opt)
    print(model.layers[-2].get_output_at(0))

    return model


def convert_to_normal(model, num_classes):
    print("model layers")
    print(model.layers)
    new_model = Sequential()
    for layer in model.layers[1:3]:
        new_model.add(layer)

    for layer in new_model.layers:
        layer.trainable = False
    if num_classes:
        new_model.add(Dense(output_dim=num_classes, activation='elu'))
    model = new_model

    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-4)

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model


def initialize_datagen(x_train):
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=0,
        # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,
        # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        # preprocessing_function=lambda t: random.choice([t, 1 - t]), # randomly invert images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    return datagen


def word_datagen(datagen, x, x_words, y, batch_size):
    generator = ImageDataGenerator()
    normal = datagen.flow(x, y, shuffle=False, batch_size=batch_size)
    original_shape = x_words.shape
    x2 = generator.flow(x_words.reshape(original_shape + (1, 1)), shuffle=False, batch_size=batch_size)
    while True:
        x1, y1 = normal.next()
        words = x2.next()
        if words.shape[0] != batch_size:
            print(words.shape)
            continue
        yield [x1, words.reshape((batch_size, original_shape[1]))], y1


def train_model(model, datagen, x_train, y_train, x_test, y_test, batch_size, epochs, train_words=None, test_words=None):
    if train_words is not None and test_words is not None:
        model_info = model.fit_generator(word_datagen(datagen, x_train, train_words, y_train, batch_size),
                                         steps_per_epoch=x_train.shape[0] // batch_size,
                                         epochs=epochs,
                                         validation_data=word_datagen(datagen, x_test, test_words, y_test, len(x_test)).next(),
                                         workers=1)
        return model_info

    model_info = model.fit_generator(datagen.flow(x_train, y_train,
                                                  batch_size=batch_size),
                                     steps_per_epoch=x_train.shape[0] // batch_size,
                                     epochs=epochs,
                                     validation_data=datagen.flow(x_test,
                                                                  y_test,
                                                                  batch_size=len(x_test)).next(),
                                     workers=4)

    return model_info


def make_buckets(x, y):
    ys = np.unique(y)
    print(ys)
    y = y.flatten()
    buckets = {int(c): x[y == c] for c in ys}
    for bucket in buckets:
        print(buckets[bucket].shape)
    return buckets


def pick_example(buckets, positive, classification):
    if positive:
        return random.choice(buckets[classification])
    options = buckets.keys()
    options.remove(classification)
    return random.choice(buckets[random.choice(options)])
    

def train_siamese_model(model, datagen, x_train, y_train, x_test, y_test, batch_size, epochs):
    buckets = make_buckets(x_train, np.argmax(y_train, axis=1))

    def generator(x, y, buckets):
        positives = [1 for i in range(batch_size / 2)] + [0 for i in range(batch_size / 2)]
        positives = [i % 2 for i in range(batch_size)]
        while True:
            px, py = datagen.flow(
                x, y, batch_size=batch_size).next()
            bla = np.array([pick_example(buckets, example, np.argmax(py[idx])) for idx, example in enumerate(positives)])
            yield ([px, bla], np.array(positives))

    model_info = model.fit_generator(generator(x_train, y_train, buckets),
                                     steps_per_epoch=x_train.shape[0] // batch_size,
                                     epochs=epochs,
                                     workers=1)

    return model_info


def save_model(model, save_dir, epochs, embedding_size):
    model_path = os.path.join(save_dir, "small_cnn_weights_{}_{}.h5".format(epochs, embedding_size))
    model.save(model_path)
    model_json_path = os.path.join(save_dir, 'small_cnn_{}.json'.format(embedding_size))

    with open(model_json_path, 'w') as outfile:
        outfile.write(model.to_json())

    print('Saved trained model at {}'.format(model_path))


def save_confusion_matrix(data_type, datagen, model, x, y, class_names, cnn_params, words=None, anomaly=False, exp_name=''):
    """:param data_type: string representing 'training' or 'test'"""
    embedding_size = cnn_params["embedding_size"]
    epochs = cnn_params["epochs"]
    save_dir = cnn_params["save_dir"]

    file_path = os.path.join(save_dir, 'confusion_{}_{}_{}.png'.format(data_type,
                                                                       epochs,
                                                                       embedding_size))
    acc, y_pred, y_true, datagen_x = accuracy(x,
                                              y,
                                              model,
                                              class_names,
                                              file_path,
                                              datagen,
                                              words,
                                              anomaly)

    results = [
        "Accuracy on {} data is: {:0.2f}".format(data_type, acc),
        "Macro precision",
        metrics.precision_score(y_true, y_pred, average='macro'),
        "Macro recall",
        metrics.recall_score(y_true, y_pred, average='macro')
    ]

    results_string = "\n".join(str(result) for result in results)
    print(results_string)
    with open(os.path.join(save_dir, 'results_{}.txt'.format(data_type)), 'w') as results_file:
        results_file.write(results_string)


def write_images(x, y, y_true, class_names, data_type, save_dir, exp_name=''):
    path = settings.PATHS['icons']

    metadata = {
        "data": {},
        "metadata": {
            "type": "classification_result",
            "subfolder": data_type,
            "exp": exp_name or save_dir,
        },
    }

    precisions = {label: metrics.precision_score(y_true, y, average='micro', labels=[idx])
                  for idx, label in enumerate(class_names)}
    recalls = {label: metrics.recall_score(y_true, y, average='micro', labels=[idx])
               for idx, label in enumerate(class_names)}

    for idx, row in enumerate(x):
        image = row.reshape(32, 32)
        class_name = class_names[int(y[idx])]
        image_folder = os.path.join(path, metadata["metadata"]["exp"], data_type, class_name)
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "{}.png".format(idx))
        imsave(image_path, image)
        metadata["data"][class_name] = metadata["data"].get(class_name,
                                                            {
                                                                "resource_id": class_name,
                                                                "top_words": [],
                                                                "closest_training_icons": [],
                                                                "precision": precisions[class_name],
                                                                "recall": recalls[class_name],
                                                            })
        metadata["data"][class_name]["closest_training_icons"] += [idx]

    return metadata


def generate_embeddings(model, x_train):
    embedding_model = Sequential()
    for layer in model.layers:
        embedding_model.add(layer)
        if layer.name == 'embedding':
            break
    embeddings = embedding_model.predict(x_train)
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        help=("Full path to the h5 model file"))
    parser.add_argument("--save_images",
                        help="Where to save the test image classifications")
    parser.add_argument("--siamese", action='store_true', help='Siamese or not?')
    parser.add_argument("--words", action='store_true', help='Words or not?')
    parser.add_argument("--visualize", help='Want to just visualize the model? Put a path to where you want the image!')
    parser.add_argument("--model_type", help='Simple or not?')
    parser.add_argument("--save_dir", help='Subfolder to save images in')
    parser.add_argument("--embeddings", action='store_true', help='Generate embeddings?')
    parser.add_argument("--anomaly", action='store_true', help='Measure accuracy with anomaly detection')
    args = parser.parse_args()

    assert os.path.isdir('../../data/icon_crops')

    np.random.seed(settings.RANDOM_SEED)
    random.seed(settings.RANDOM_SEED)
    cnn_params = settings.CNN_PARAMS
    embedding_size = cnn_params["embedding_size"]
    batch_size = cnn_params["batch_size"]
    epochs = cnn_params["epochs"]
    save_dir = args.save_dir or cnn_params["save_dir"]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    x_train, x_test, y_train, y_test, num_classes = load_data(settings.data_folder)

    train_words = test_words = None
    if args.words:
        train_words, test_words = load_word_data(settings.data_folder)

    datagen = initialize_datagen(x_train)

    with open(os.path.join(save_dir, 'datagen.pkl'), 'wb') as datagen_file:
        pickle.dump(datagen, datagen_file)

    if not args.model_path:
        model = initialize_model(cnn_params["embedding_size"], num_classes, args.model_type)

        start_time = time.time()
        model_info = train_model(model, datagen, x_train, y_train, x_test, y_test, batch_size, epochs)
        end_time = time.time()

        print("Model took %0.2f minutes to train" % ((end_time - start_time) / 60.0))
    else:
        model = load_model(args.model_path)

    if args.embeddings:
        generate_embeddings(model, x_train, y_train, 'train', save_dir, datagen)
        return

    if args.visualize:
        plot_model(model, to_file=args.visualize, show_shapes=True)
        return

    with open(os.path.join(settings.data_folder("validation_metadata.json")), 'r') as infile:
        class_names = json.load(infile)["class_names"]

    save_confusion_matrix("train", datagen, model, x_train, y_train, class_names, cnn_params, train_words, args.anomaly, args.save_images)
    save_confusion_matrix("test", datagen, model, x_test, y_test, class_names, cnn_params, test_words, args.anomaly, args.save_images)

    img_paths = glob.glob('../../data/icon_crops/*')
    loaded = []
    for i in range(0, len(img_paths)):
        if i % 1000 == 0:
            print(i)

        load_img = np.array(Image.open(img_paths[i]).convert('L')).astype('float32')
        my_img = np.expand_dims(load_img, 2) / 255
        loaded.append(my_img)
    print('Loading done... passing through model now')
    all_embeds = []
    for j in range(0, len(loaded), 2000):
        print(j)
        batch = np.array(loaded[j:j+2000])
        embeds = generate_embeddings(model, batch)
        all_embeds.append(embeds)
    all_embeds = [sample for batch in all_embeds for sample in batch]
    print('Number of icon embeddings = %d' % len(all_embeds))
    print('Saving npy...')
    np.save('../../data/icon_features', all_embeds)

    if not args.model_path:
        model_history_file_path = os.path.join(save_dir, "small_cnn_info_{}_{}.pdf".format(epochs,
                                                                                           embedding_size))

        save_model(model, save_dir, epochs, embedding_size)

if __name__ == "__main__":
    main()
