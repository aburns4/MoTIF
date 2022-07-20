import json
import glob
import numpy as np
import os
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model

print('Loading FastText model...')
model_path = datapath(os.path.join(os.getcwd(), 'wiki.en.bin'))
fb_model = load_facebook_model(model_path)
print('Done!')

w2i = {}
vectors = []
all_words = []

assert os.path.isdir('../data/view_hierarchy_features')
all_view_data = glob.glob('../data/view_hierarchy_features/*')
for view in all_view_data:
    with open(view) as f:
        data = json.load(f)
        for t in data['text']:
            all_words.append(t)
        for t in data['ids']:
            all_words.append(t)
        for t in data['classes']:
            all_words.append(t)

all_words = list(set(all_words))
print(len(all_words))
vectors = list(fb_model.wv[all_words])
print(len(vectors))
pad = [0.]*300 # pad vector
vectors.append(pad)
print(len(vectors))

np.save('../data/fasttext_vectors.npy', vectors)
idxs = range(len(all_words))
w2i_map = list(zip(all_words, idxs))
for entry in w2i_map:
    w2i[entry[0]] = entry[1]

with open('../data/w2i_map.json', 'w') as f:
    json.dump(w2i, f)