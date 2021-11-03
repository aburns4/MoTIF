# Semantic Icon Classifier

A CNN based model classifying 99 icon classes appear most often in Android apps. Sometimes, we have input images that do not belong to any of the 99 classes, either can be a random image or a very rarely used icon class. Therefore, we develop an extra Anomaly detector to handle this case. Our work is presented in the paper "Learning Design Semantics for Mobile Apps".

## Requirements
Python: ``2.7.12``

Create a new local environment using any package management tools (e.g. ``virtualenv``, ``Anaconda``). Then, install the following packages with the specified version.
* keras 2.0.8
* tensorflow 1.8.0
* Pillow 5.2.0
* mock 2.0.0

We provide a complete list of packages installed in our development machine `requirements.txt`.

## Download data

1. ``data.zip``: [(Download)](https://drive.google.com/open?id=1SiD_U5ifjX1poJZzLB-MwvoUQBhutYzH)


What is inside:
```
training_x.npy: training icons (requires for both train and test evaluation)

training_y.npy: training labels (requires for both train and test evaluation)

validation_x.npy: testing icons (requires for both train and test evaluation)

validation_y.npy: testing labels (requires for both train and test evaluation)

validation_metadata.json: meta information about each icon (require for both train and test evaluation)

anomalies_embeddings.npy:

gmm_invalid_class.npy:

gmm_valid_class.npy:

x_train_class.npy:

y_train_embeddings.npy:

```

2. ``saved_models.zip``: [(Download)](https://drive.google.com/open?id=16hUHUzxkGHHBRsgvfeLV_4gTD3-KFYIy)

What is inside:
```
anomaly.pkl: trained Anomaly Detection model (requires in model evaluation with anomaly detector)

inv_anomaly.pkl: trained Anomaly Detection model (requires in model evaluation with anomaly detector)

datagen.pkl: file generated during training (requires in model evaluation)
```

3. `icon.zip`: [(Download)](https://drive.google.com/file/d/1D0CFmDP0xNSyfSkK7kUHnfP0HnpcKZc1)
You do not need this zip file in training a model. In the evaluation, you may need these raw icon images when passing an argument flag  ``--save_images``  to generate a testing sample. The generated file is stored in JSON. **Warning**: Over 100k of image files, becareful aftre you unzip it.

## How to train our icon classifier

#### Step 1

Unzip `data.zip` in the current directory.

Create a folder ``saved_models``. Ignore `datagen.pkl` that we provided, this file is generated again during the training process.

#### Step 2

```
python2 cnn_pretrain.py
```

When training is finish, you will see the following message printed in the terminal:

```
Accuracy on train data is: 99.68
Macro precision
0.99665404376452
Macro recall
0.9966673171192078
...
Accuracy on test data is: 94.68
Macro precision
0.8748637458382006
Macro recall
0.8552836877294613
```
These results are evaluated without anomaly detector. Our trained model is ``small_cnn_weights_100_512.h5``[(Download)](https://drive.google.com/file/d/1Kq5agoiLSuv5_CVBlkf5F7iENtpyKyz8).


## Evaluate your icon classifier from a saved model

#### Evaluate without anomaly detection
```

python2 cnn_pretrain.py --model_path ./saved_models/small_cnn_weights_100_512.h5
```

#### Evaluate with anomaly detection
```

python2 cnn_pretrain.py --anomaly --model_path ./saved_models/small_cnn_weights_100_512.h5
```

## Notes
* You can change any hyperparameters in `settings.py`. 
* Our default CNN training epoch is `100`.

## Contributions
* Paper: [Learning Design Semantics for Mobile Apps](http://interactionmining.org/rico)

## If you have any question, please contact:
* Jason Situ (junsitu2@illinois.edu)
