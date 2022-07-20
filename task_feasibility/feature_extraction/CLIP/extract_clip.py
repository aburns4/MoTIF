import torch
import clip
import json
import glob
import os
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_path = '../../data/clip_image_features'
if not os.path.isdir(clip_path):
    os.mkdir(clip_path)

assert os.path.isfile('../../data/w2i_map.json')
with open('../../data/w2i_map.json') as f:
    w2i = json.load(f)
    vocab = list(w2i.keys())

model, preprocess = clip.load("ViT-B/32", device=device)

print('Processing text + view hierarchy vocab file...')
text = clip.tokenize(vocab).to(device)
print('Done!')

print('Extracting text features...')
all_text = torch.empty(0, 512).to(device)
with torch.no_grad():
    for i in range(0, len(text), 1000):
        print(i)
        text_features = model.encode_text(text[i:i+1000])
        all_text = torch.cat([all_text, text_features], dim=0)

pad = torch.zeros(1, 512).to(device)
all_text = torch.cat([all_text, pad], dim=0)
print('Done!')

print(all_text.size())
all_text = all_text.cpu().numpy()
np.save('../../data/clip_text_vectors.npy', all_text)

image_paths = glob.glob('../../raw_data/*/*/screens/*')
print('Extracting image features...')
with torch.no_grad():
    for i in range(0, len(image_paths), 1000):
        print(i)
        print('Processing images...')
        batch_paths = image_paths[i:i+1000]
        preprocessed_images = [preprocess(Image.open(im_path)).unsqueeze(0).to(device) for im_path in batch_paths]
        image_batch = torch.cat(preprocessed_images)
        print(image_batch.size())
        print('Done!')

        image_features = model.encode_image(image_batch)
        print(image_features.size())
        print('Writing batch to file')
        for i in range(len(batch_paths)):
            write_to = os.path.join(clip_path, batch_paths[i].split('/')[-1].split('.')[0])
            sample_feat = image_features[i].clone()
            torch.save(sample_feat, write_to)
        print('Done!')