import argparse
import json
import torch
import glob
import os
from Screen2Vec import Screen2Vec
from sentence_transformers import SentenceTransformer
from prediction import TracePredictor
from autoencoder import ScreenLayout, LayoutAutoEncoder
from UI_embedding.UI2Vec import HiddenLabelPredictorModel, UI2Vec
from dataset.rico_utils import get_all_labeled_uis_from_rico_screen, ScreenInfo
from dataset.rico_dao import load_rico_screen_dict

# Generates the vector embeddings for an input screen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
parser = argparse.ArgumentParser()

parser.add_argument("-u", "--ui_model", required=True, type=str, help="path to UI embedding model")
parser.add_argument("-n", "--num_predictors", type=int, default=4, help="number of other labels used to predict one")
parser.add_argument("-m", "--screen_model", required=True, type=str, help="path to screen embedding model")
parser.add_argument("-l", "--layout_model", required=True, type=str, help="path to layout embedding model")
parser.add_argument("-v", "--net_version", type=int, default=4, help="0 for regular, 1 to embed location in UIs, 2 to use layout embedding, 3 to use both, 4 with both but no description, 5 to use both but not train description")

args = parser.parse_args()


def get_embedding(screens, ui_model, screen_model, layout_model, num_predictors, net_version):
    bert = SentenceTransformer('bert-base-nli-mean-tokens')
    bert_size = 768

    loaded_ui_model = HiddenLabelPredictorModel(bert, bert_size, 16)
    loaded_ui_model.load_state_dict(torch.load(ui_model), strict=False)

    layout_autoencoder = LayoutAutoEncoder()
    layout_autoencoder.load_state_dict(torch.load(layout_model))
    layout_embedder = layout_autoencoder.enc

    if net_version in [0,2,6]:
        adus = 0
    else:
        # case where coordinates are part of UI rnn
        adus = 4
    if net_version in [0, 1, 6]:
        adss = 0
    else:
        # case where screen layout vec is used
        adss = 64
    if net_version in [0, 1, 2, 3]:
        desc_size = 768
    else:
        # no description in training case
        desc_size = 0

    screen_embedder = Screen2Vec(bert_size, adus, adss, net_version)
    loaded_screen_model = TracePredictor(screen_embedder, net_version)
    loaded_screen_model.load_state_dict(torch.load(screen_model))

    screen_embeddings = []
    issues = 0
    done = 0
    for screen in screens:
        if done % 1000 == 0:
            print(done)
        done+=1
        try:
            with open(screen) as f:
                rico_screen = load_rico_screen_dict(json.load(f))
        except:
            issues += 1
            screen_embeddings.append(torch.randn(768))
            continue

        labeled_text = get_all_labeled_uis_from_rico_screen(rico_screen)

        ui_class = torch.tensor([UI[1] for UI in labeled_text])
        ui_text = [UI[0] for UI in labeled_text]
        if len(ui_class) == 0 and len(ui_text) == 0:
            screen_embeddings.append(torch.randn(768))
            issues += 1
        else:
            UI_embeddings = loaded_ui_model.model([ui_text, ui_class])
            descr = ''
            descr_emb = torch.as_tensor(bert.encode([descr]), dtype=torch.float)

            screen_to_add = ScreenLayout(screen)
            screen_pixels = screen_to_add.pixels.flatten()
            encoded_layout = layout_embedder(torch.as_tensor(screen_pixels, dtype=torch.float).unsqueeze(0)).squeeze(0)

            if net_version in [1,3,4,5]:
                coords = torch.FloatTensor([labeled_text[x][2] for x in range(len(UI_embeddings))])
                UI_embeddings = torch.cat((UI_embeddings.cpu(), coords), dim=1)
            if net_version in [0,1,6]:
                screen_layout = None
            else:
                screen_layout = encoded_layout.unsqueeze(0).unsqueeze(0)

            screen_emb = screen_embedder(UI_embeddings.unsqueeze(1).unsqueeze(0), descr_emb.unsqueeze(0), None, screen_layout, False)
            screen_embeddings.append(screen_emb.cpu())
    return screen_embeddings

s2v_path = '../../data/screen2vec_features'
if not os.path.isdir(s2v_path):
    os.mkdir(s2v_path)

all_screens = glob.glob('../../raw_data/*/*/view_hierarchies/*')
print(len(all_screens))
trace_ids = [x.split('/')[-1] for x in all_screens]

screen_embeds = get_embedding(all_screens, args.ui_model, args.screen_model, args.layout_model, args.num_predictors, args.net_version)
print('Got the embeddings! Write to file now')
for i in range(len(screen_embeds)):
    if i % 1000 == 0:
        print(i)
    embed = screen_embeds[i]
    final_embed = torch.squeeze(embed).clone()
    write_to = '../../data/screen2vec_features/' + trace_ids[i]
    torch.save(final_embed, write_to)
