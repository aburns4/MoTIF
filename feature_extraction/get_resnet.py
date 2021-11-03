import glob
import torch
import torch.nn as nn
import torchvision.models as models
import os
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

resnet_path = '../data/resnet_features'
if not os.path.isdir(resnet_path):
    os.mkdir(resnet_path)

def get_paths(base_dir):
    paths = glob.glob(base_dir + '/*/*/screens/*')
    return paths

def get_crops_and_resize(img, transform):
    # input it single img of varying dim
    # gets 4 crops given input img dims
    # resizes each to 224 x 224 for resnet
    width, height = img.size
    c_and_r = []

    crop1 = (0, 0, int(width/2), int(height/2))
    crop2 = (int(width/2), 0, width, int(height/2))
    crop3 = (int(width/2), int(height/2), width, height)
    crop4 = (0, int(height/2), int(width/2), height)
    crop5 = (int(width/4), int(height/4), int(3*width/4), int(3*height/4))

    crops = [crop1, crop2, crop3, crop4, crop5, crop1, crop2, crop3, crop4, crop5]
    for i in range(len(crops)):
        crop_i = crops[i]
        if i > 4:
            # do not flip one set of crops
            im_cr = img.crop(crop_i).resize((224,224))
        else:
            im_cr = img.crop(crop_i).resize((224,224))
            im_cr = im_cr.transpose(Image.FLIP_LEFT_RIGHT)

        im_cr = transform(im_cr)
        c_and_r.append(im_cr)
    return torch.stack(c_and_r)

# set up resnet
resnet152 = models.resnet152(pretrained=True).to(device)
modules = list(resnet152.children())[:-1]
resnet152 = nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False

# get batch of images
paths = get_paths('../raw_data/')
total = 0
written = []
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])
for idx in range(0, len(paths), 50):
    if idx % 1000 == 0:
        print('View path %d' % idx)
        
    batch_feats = torch.empty(0, 3, 224, 224).to(device)
    resnet_paths = []
    for img_path in paths[idx:idx+50]:
        resnet_id = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path)
        mini_batch = get_crops_and_resize(img, transform).to(device)
        for j in range(mini_batch.size()[0]):
            total += 1
            resnet_paths.append(resnet_id + '.' + str(j))
        batch_feats = torch.cat([batch_feats, mini_batch], dim=0)
        batch_feats = batch_feats.to(device)
        img.close()

    # get the output from the last hidden layer of the pretrained resnet
    features_var = resnet152(batch_feats)
    # get the tensor out of the variable
    features = torch.squeeze(features_var.data)
    print('Writing to file')
    for i in range(len(resnet_paths)):
        write_to = '../data/resnet_features/' + resnet_paths[i]
        written.append(write_to)
        sample_feat = features[i].clone()
        torch.save(sample_feat, write_to)
    print('Done!')

print('Total processed = %d' % total)
print('Total written length = %d' % len(written))