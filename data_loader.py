import json

from numpy.lib.shape_base import split
import torch
import os
import pickle as p
import math
import numpy as np
import glob
from utils import convert_class, pad_view, pad_icon_feats, pad_sent, count_frequency

class DataLoader():
    def __init__(self, configs) -> None:
        self.data_dir = configs['data_dir']
        self.num_views = configs['views']
        self.num_crops = configs['num_crops']
        self.which_view = configs['which_view']
        self.vis_feat_info = configs['vis_feat_dims']
        self.word2index_path = configs['word2index_path']
        self.vocab_path = configs['vocab_path']
        self.use_view_text = configs['use_view_text'] 
        self.use_view_id = configs['use_view_ids'] 
        self.use_view_cls = configs['use_view_cls']
        self.split = configs['split']
        self.annotation_path = configs['annotation_path']

        max_view_text_toks = 0
        if self.use_view_text:
            max_view_text_toks += 100
        if self.use_view_id:
            max_view_text_toks += 50
        if self.use_view_cls:
            max_view_text_toks += 50
        self.max_view_text_toks = max_view_text_toks

        self.load_meta()

    def load_meta(self):
        '''
        Loads list of samples, the word to index dictionary mapping,
        and the word embedding vocab matrix.
        '''
        split_path = os.path.join(self.data_dir, self.split + '_traces.txt')
        annotation_path = os.path.join(self.data_dir, self.annotation_path)
        w2i_path = os.path.join(self.data_dir, self.word2index_path)
        voc_mat_path = os.path.join(self.data_dir, self.vocab_path)

        with open(split_path) as f:
            split_traces = f.readlines()
            split_traces = [x.strip() for x in split_traces]

        keep_anns = []
        with open(annotation_path, 'rb') as f:
            anns = p.load(f)
            for ann in anns:
                if ann[1] in split_traces:
                    keep_anns.append(ann)
        print(len(keep_anns))
        with open(w2i_path) as f:
            word2index = json.load(f)

        voc_matrix = np.load(voc_mat_path)

        self.annotations = keep_anns
        self.word2index = word2index
        self.vocab = voc_matrix

    def views_to_keep(self, view_paths):
        ''' Preprocesses view paths given the experiment config.
        Args:
            view_paths: [[a sample's view paths]]
        Returns: 
            views_to_load: subset of each sample's view_paths to load; 
                           can be 1 view, 3 views, or n views as defined                                   
                           in config
        '''
        if self.num_views == 3:
            # load first, mid, and last view
            mid_idxs = [math.floor(len(x) / 2) for x in view_paths]
            views_to_load = []
            for i in range(len(view_paths)):
                vp = view_paths[i]
                views_to_load.append([vp[0], vp[mid_idxs[i]], vp[-1]])
        elif self.num_views == 1:
            # load only one view
            if self.which_view == 'FIRST':
                views_to_load = [[x[0]] for x in view_paths]
            elif self.which_view == 'MID':
                views_to_load = []
                mid_idxs = [math.floor(len(x) / 2) for x in view_paths]
                for i in range(len(view_paths)):
                    vp = view_paths[i]
                    views_to_load.append([vp[mid_idxs[i]]])
            else:
                # which_view == LAST
                views_to_load = [[x[-1]] for x in view_paths]
        else:
            # load (up to) n views
            views_to_load = [v[:self.num_views] for v in view_paths]
        return views_to_load


    def build_icon_matrix(self, views_to_load, all_icon_paths, all_embeds):
        ''' Loads icon features per view.
        Args:
            views_to_load: processed views to keep per sample, either 
                one view, three views, or n views per trace as defined
                in the config.
            all_icon_paths: paths to all views' icon features, needed
                to identify view id.
            all_embeds: icon embeddings for all views.
        Returns:
            matrix: a dictionary of {view_id: [icon embeddings]} key,
                value pairs where each view has a list of 512D icon
                representations.
        '''
        matrix = {}
        for i in range(len(all_icon_paths)):
            view_id = all_icon_paths[i].split('/')[-1].split('.')[0]
            if view_id in matrix:
                matrix[view_id].append(list(all_embeds[i]))
            else:
                matrix[view_id] = [list(all_embeds[i])]

        for trace in views_to_load:
            for view in trace:
                if view[:-4] not in matrix:
                    matrix[view[:-4]] = [list(np.random.normal(0, 1, 512))]
        return matrix


    def load_view_feats(self, views_to_load, feature_type):
        ''' Loads ResNet, Screen2Vec, or Icon features.
        Inputs:
            views_to_load: [[a sample's preprocessed view paths]]
            feature_type: string in {ResNet, Screen2Vec, or Icon}
        Returns:
            view_feats: list of features with shape 
                        [# views, # feats per view, feat dim]
        '''
        print(feature_type)
        if feature_type == 'RESNET':
            load_path = os.path.join(self.data_dir, 'resnet_features')
            view_feats = [[[list(torch.load(os.path.join(load_path,
                                                    view[:-4] + '.' + str(crop) )).cpu().numpy()) 
                                                    for crop in range(self.num_crops)] 
                                                for view in trace] 
                                            for trace in views_to_load]
        elif feature_type == 'SCREEN2VEC':
            load_path = os.path.join(self.data_dir, 'screen2vec_features')
            view_feats = [[list(torch.load(os.path.join(load_path, view)).detach().numpy()) for view in trace] for trace in views_to_load]
        elif feature_type == 'CLIP':
            load_path = os.path.join(self.data_dir, 'clip_image_features')
            view_feats = [[list(torch.load(os.path.join(load_path, view.split('.')[0])).detach().cpu().numpy()) for view in trace] for trace in views_to_load]
        else:
            # ICON
            all_icon_paths =  glob.glob(os.path.join(self.data_dir, 'icon_crops/*'))
            all_embeds = np.load(os.path.join(self.data_dir, 'icon_features.npy'))
            icon_embeddings = self.build_icon_matrix(views_to_load, all_icon_paths, all_embeds)

            view_feats = [[icon_embeddings[view[:-4]] for view in trace] for trace in views_to_load]
            icon_lens = [len(y) for x in view_feats for y in x]
            max_icons = 57 # max(icon_lens)
            # print(max_icons)
            view_feats = pad_icon_feats(view_feats, max_icons)
        return view_feats


    def load_view_text_feats(self, views_to_load):
        ''' Loads view hierarchy text features, which can include text tags,
            element ids, and element classes. 

        Inputs:
            views_to_load: [[a sample's preprocessed view paths]]
        Returns: 
            trace_fts: list of text tokens with shape 
                       [# views, # feats per view]
        '''

        trace_fts = []
        for trace in views_to_load:
            view_fts = []
            for vp in trace:
                open_p = os.path.join(self.data_dir, 'view_hierarchy_features', vp[:-4])
                try:
                    with open(open_p) as f:
                        data = json.load(f)
                        toks = []
                        if self.use_view_text:
                            for t in data['text'][:100]:
                                toks.append(t.lower())
                        if self.use_view_id:
                            for t in data['ids'][:50]:
                                toks.append(t.lower())
                        if self.use_view_cls:
                            for t in data['classes'][:50]:
                                toks.append(t.lower())
                except:
                    toks = []
                view_fts.append(toks)
            trace_fts.append(view_fts)
        return trace_fts


    def load_features(self):
        ''' Generates all features needed for training.

        Returns:
            padded_num_tasks: tokenized sentences which have
                been mapped to vocab indices and are padded to the
                maximum tokenized sentence length.
            padded_view_text_num_tasks_over_view: padded_num_tasks
                which have been padded over the view dimension such
                that all traces have the same dimensions.
            all_vis_feats: visual features padded over view dimension
                for the features included in the config.
            feas: binary class annotations for all samples on
                whether the task was feasible or infeasible.
        '''
        tasks = [x[2] for x in self.annotations]
        feas = convert_class([x[3] for x in self.annotations])

        view_pths = [x[4] for x in self.annotations]
        kept_views = self.views_to_keep(view_pths)
        view_text = self.load_view_text_feats(kept_views)
        view_text_num_tasks = []
        for trace in view_text:
            t_idxs = []
            for view in trace:
                v_idxs = []
                for tok in view:
                    try:
                        v_idxs.append(self.word2index[tok])
                    except:
                        continue
                t_idxs.append(v_idxs)
            view_text_num_tasks.append(t_idxs)

        padded_view_text_num_tasks = [[pad_sent(view_text, self.max_view_text_toks, self.vocab.shape[0] - 1) for view_text in trace_text] for trace_text in view_text_num_tasks]
        padded_view_text_num_tasks_over_view = [np.array(pad_view(v, self.num_views, self.max_view_text_toks, 'TEXT', self.vocab.shape[0] - 1)) for v in padded_view_text_num_tasks]
        num_tasks = []
        task_lens = []
        for task in tasks:
            t_idxs = []
            toks = task.split(' ')
            for tok in toks:
                try:
                    t_idxs.append(self.word2index[tok])
                except:
                    continue
            task_lens.append(len(t_idxs))
            num_tasks.append(t_idxs)

        padded_num_tasks = [np.array(pad_sent(task, max(task_lens), self.vocab.shape[0] - 1, True))  for task in num_tasks]

        all_vis_feats = {}
        for feat_type in self.vis_feat_info:
            print('Loading view image features...')            
            views = self.load_view_feats(kept_views, feat_type)
            print('Done!')
            print('Padding view features...')
            padded_views = [np.array(pad_view(v, self.num_views, self.vis_feat_info[feat_type], feat_type)) for v in views]
            print('Done!')
            all_vis_feats[feat_type] = padded_views
            
        return padded_num_tasks, padded_view_text_num_tasks_over_view, all_vis_feats, feas
