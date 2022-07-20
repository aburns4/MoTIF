import argparse
import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import json

from data_loader import DataLoader
from model import ComboNet, LangNet, VisNet
from trainer import ComboTrainer, VisTrainer, LangTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", help="directory containing feature files",
                    default="data")
parser.add_argument("-e", "--exp_dir", help="directory to save models to",
                    default="experiments")
parser.add_argument("-cd", "--config_dir", help="directory containing model configs",
                    default="configs")
parser.add_argument("-w2i", "--word2index_path", help="file containing word to index map",
                    default="w2i_map.json")
parser.add_argument("-v", "--vocab_fasttext_path", help="file containing fasttext embeddings",
                    default="fasttext_vectors.npy")
parser.add_argument("-a", "--annotation_path", help="file containing annotations",
                    default="feasibility_annotations.p")               
parser.add_argument("-c", "--config", help="model config to use",
                    default="combo_rn_s2v_v20_lstm.json")
args = parser.parse_args()

with open(os.path.join(args.config_dir, args.config)) as f:
    exp_cfgs = json.load(f)

cmd_cfgs = {'data_dir': args.data_dir,
            'word2index_path': args.word2index_path,
            'vocab_path': args.vocab_fasttext_path,
            'annotation_path': args.annotation_path}

all_cfgs = exp_cfgs.copy()
all_cfgs.update(cmd_cfgs)
print(all_cfgs)

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 126
# convert data to torch.FloatTensor
transform = transforms.ToTensor()

train_cfgs = all_cfgs.copy()
train_cfgs.update({'split': 'train'})
TRAIN_DL = DataLoader(train_cfgs)
train_text_data, train_view_text_data, train_vis_view_data, train_labels = TRAIN_DL.load_features()

test_cfgs = all_cfgs.copy()
test_cfgs.update({'split': 'test'})
TEST_DL = DataLoader(test_cfgs)
test_text_data, test_view_text_data, test_vis_view_data, test_labels = TEST_DL.load_features()

if all_cfgs['net'] == 'combo':
    train_text_data = torch.from_numpy(np.array(train_text_data))
    test_text_data = torch.from_numpy(np.array(test_text_data))

    f1 = list(train_cfgs['feat_dims'].keys())[0]
    f2 = list(train_cfgs['feat_dims'].keys())[1]
    if f1 in ['RESNET', 'ICON', 'SCREEN2VEC', 'CLIP']:
        train_feat_1_data = train_vis_view_data[f1]
        train_feat_1_data = torch.from_numpy(np.array(train_feat_1_data)).float()

        test_feat_1_data = test_vis_view_data[f1]
        test_feat_1_data = torch.from_numpy(np.array(test_feat_1_data)).float()
    else:
        # view hierarchy features
        train_feat_1_data = torch.from_numpy(np.array(train_view_text_data))
        test_feat_1_data = torch.from_numpy(np.array(test_view_text_data))

    if f2 in ['RESNET', 'ICON', 'SCREEN2VEC', 'CLIP']:
        train_feat_2_data = train_vis_view_data[f2]
        train_feat_2_data = torch.from_numpy(np.array(train_feat_2_data)).float()

        test_feat_2_data = test_vis_view_data[f2]
        test_feat_2_data = torch.from_numpy(np.array(test_feat_2_data)).float()
    else:
        # view hierarchy features
        train_feat_2_data = torch.from_numpy(np.array(train_view_text_data))
        test_feat_2_data = torch.from_numpy(np.array(test_view_text_data))

    # initialize the NN
    model = ComboNet(all_cfgs).to(device)
    print(model)
    motif_train_data = list(zip(train_text_data, train_feat_1_data, train_feat_2_data, train_labels))
    motif_test_data = list(zip(test_text_data, test_feat_1_data, test_feat_2_data, test_labels))
    print(len(motif_train_data))
    print(len(motif_test_data))

    motif_train_loader = torch.utils.data.DataLoader(motif_train_data, batch_size=batch_size,
                                                    num_workers=num_workers)
    motif_test_loader = torch.utils.data.DataLoader(motif_test_data, batch_size=batch_size,
                                                    num_workers=num_workers)
    ## Specify loss and optimization functions
    # specify loss function
    criterion = nn.CrossEntropyLoss()   
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = ComboTrainer(motif_train_loader, motif_test_loader, model, criterion, optimizer, device, args.config)
elif all_cfgs['net'] == 'vis':
    train_text_data = torch.from_numpy(np.array(train_text_data))
    assert len(list(train_vis_view_data.keys())) == 1
    train_vis_feat = list(train_vis_view_data.keys())[0]
    train_vis_data = train_vis_view_data[train_vis_feat]
    train_vis_data = torch.from_numpy(np.array(train_vis_data)).float()

    test_text_data = torch.from_numpy(np.array(test_text_data))
    assert len(list(test_vis_view_data.keys())) == 1
    test_vis_feat = list(test_vis_view_data.keys())[0]
    test_vis_data = test_vis_view_data[train_vis_feat]
    test_vis_data = torch.from_numpy(np.array(test_vis_data)).float()

    # initialize the NN
    model = VisNet(all_cfgs).to(device)
    print(model)
    motif_train_data = list(zip(train_text_data, train_vis_data, train_labels))
    motif_test_data = list(zip(test_text_data, test_vis_data, test_labels))
    print(len(motif_train_data))
    print(len(motif_test_data))

    motif_train_loader = torch.utils.data.DataLoader(motif_train_data, batch_size=batch_size,
                                                    num_workers=num_workers)
    motif_test_loader = torch.utils.data.DataLoader(motif_test_data, batch_size=batch_size,
                                                    num_workers=num_workers)
    ## Specify loss and optimization functions
    # specify loss function
    criterion = nn.CrossEntropyLoss()   
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = VisTrainer(motif_train_loader, motif_test_loader, model, criterion, optimizer, device, args.config)
else:
    # lang
    train_text_data = torch.from_numpy(np.array(train_text_data))
    train_view_text_data = torch.from_numpy(np.array(train_view_text_data))

    test_text_data = torch.from_numpy(np.array(test_text_data))
    test_view_text_data = torch.from_numpy(np.array(test_view_text_data))
    # initialize the NN
    model = LangNet(all_cfgs).to(device)
    print(model)
    motif_train_data = list(zip(train_text_data, train_view_text_data, train_labels))
    motif_test_data = list(zip(test_text_data, test_view_text_data, test_labels))
    print(len(motif_train_data))
    print(len(motif_test_data))

    motif_train_loader = torch.utils.data.DataLoader(motif_train_data, batch_size=batch_size,
                                                    num_workers=num_workers)
    motif_test_loader = torch.utils.data.DataLoader(motif_test_data, batch_size=batch_size,
                                                    num_workers=num_workers)
    ## Specify loss and optimization functions
    # specify loss function
    criterion = nn.CrossEntropyLoss()   
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = LangTrainer(motif_train_loader, motif_test_loader, model, criterion, optimizer, device, args.config)

trainer.train()
trainer.eval()