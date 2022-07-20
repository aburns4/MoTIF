import torch
import numpy as np
import collections


def count_frequency(my_list):
    ''' Creates a count dictionary given a list
        of items.
    '''
    freq = {}
    for item in my_list:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    return collections.OrderedDict(sorted(freq.items()))


def convert_class(str_classes):
    ''' Converts string feasibility labels
        into integer binary labels.
    '''
    int_classes = []
    for label in str_classes:
        if label == 'Yes':
            int_classes.append(1)
        else:
            int_classes.append(0)
    return np.array(int_classes)


def pad_sent(sent, max_toks, idx_to_pad, arr=False):
    ''' Pad each tokenized sentence to the maximum
        number of tokens.
    '''
    while len(sent) < max_toks:
        sent.append(idx_to_pad)
    if arr:
        return np.array(sent)
    else:
        return sent

def pad_icon_feats(icon_feats, pad_len):
    ''' Pad each view within a trace to the 
        maximum number of icon features.
    '''
    for tr in icon_feats:
        for vi in tr:
            while len(vi) < pad_len:
                vi.append([0] * 512)
    return icon_feats


def pad_view(view, max_views, feat_size, dtype, tok_to_pad=None):
    ''' Pad each trace's features view dimension to the max
        number of views.
    '''
    while len(view) < max_views:
        if dtype == 'RESNET':
            crop = [0] * feat_size
            view.append(10 * [crop])
        elif dtype == 'TEXT':
            view.append([tok_to_pad] * feat_size)
        elif dtype == 'SCREEN2VEC' or dtype == 'CLIP':
            view.append([0] * feat_size)
        else:
            icon = [0] * feat_size
            view.append(57 * [icon])
    return view

def normalize(feats):
    ''' Normalize over the feature dimension in a batch.
    '''
    mu = torch.mean(feats, dim=0)
    std = torch.std(feats, dim=0)
    normalized_feats = (feats - mu) / std
    return normalized_feats