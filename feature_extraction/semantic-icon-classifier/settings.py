import os

data_folder_path = './data'
data_folder = lambda path, *paths: os.path.join(data_folder_path, path, *paths)
NUM_CONSTRAINTS = 10000000
RANDOM_SEED = 1000
CNN_PARAMS = {
    "embedding_size": 512,
    "batch_size": 32,
    "epochs": 100,
    "num_predictions": 0,
    "save_dir": "saved_models",
}
MODEL_PARAMS = {
    "outer_loops": 1,
    "num_clusters": 200,
    "pckmeans_iters": 50,
    "cnn_epochs": 20,
    "cnn_lr": 0.00001,
    "pos_violations_weight": 40000000 / 1000,
}
PATHS = {
    "icons": data_folder("icons"),
    "temp_weights": os.path.join(CNN_PARAMS["save_dir"], "temp"),
}
CLASSES_TO_MERGE = {
    "time": ["access_time", "alarm"],
    "more": ["more_horizontal", "more_vertical"],
    "flight": ["flight", "airplane_mode"],
    "playlist": ["playlist", "queue_music"],
    "warning": ["warning", "error", "announcement"],
    "clipboard": ["paste", "assignment"],
}
