import os
import torch
import torch.nn as nn
import numpy as np
from utils import normalize

class VisNet(nn.Module):
    def __init__(self, configs):
        super(VisNet, self).__init__()
        self.text_dims = configs['text_dims']
        self.max_views = configs['views']
        self.join = configs['join_type']
        self.crop_join = configs['crop_join']
        self.num_crops = configs['num_crops']
        self.vis_feats = list(configs['vis_feat_dims'].keys())[0]
        assert len(list(configs['vis_feat_dims'].keys())) == 1
        self.vocab_size = configs['vocab_size']
        self.embed = nn.Embedding(self.vocab_size, self.text_dims)
        self.embed.load_state_dict({'weight': torch.from_numpy(np.load(os.path.join(configs['data_dir'], configs['vocab_path'])))})

        if self.vis_feats == 'RESNET':
            if self.crop_join == 'AVG':
                self.vis_feat_dims = 2048
            else:
                self.vis_feat_dims = 2048 * self.num_crops
        elif self.vis_feats == 'SCREEN2VEC':
            self.vis_feat_dims = 768
        else:
            # ICON
            if self.crop_join == 'AVG':
                self.vis_feat_dims = 512
            else:
                self.vis_feat_dims = 512 * self.num_crops

        if self.join == 'LSTM':
            self.lstm = nn.LSTM(self.vis_feat_dims, 512, batch_first=True)
            self.vis_dims = 512
        elif self.join == 'AVG':
            self.vis_dims = self.vis_feat_dims
        else:
            # CONCAT
            self.vis_dims = self.max_views * self.vis_feat_dims

        self.fc1 = nn.Linear(self.text_dims + self.vis_dims, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_text, x_vis):
        # embed text features
        text_embed = torch.mean(self.embed(x_text), dim=1)

        # join vis features
        if self.vis_feats == 'RESNET' or self.vis_feats == 'ICON':
            if self.crop_join == 'AVG':
                x_vis = torch.mean(x_vis, dim=2)
            else:
                # concat
                x_vis = torch.flatten(x_vis, start_dim=2)

        if self.join == 'LSTM':
            _, (hidden_state, _) = self.lstm(x_vis)
            vis_embed = torch.squeeze(hidden_state)
        elif self.join == 'AVG':
            vis_embed = torch.squeeze(torch.mean(x_vis, dim=1))
        else:
            # concat
            vis_embed = torch.flatten(x_vis, start_dim=1)

        input_feats = normalize(torch.cat([text_embed, vis_embed], dim=1))
        hidden = self.relu(self.fc1(input_feats))
        hidden2 = self.relu(self.fc2(hidden))
        output = self.sigmoid(self.fc3(hidden2))

        return output


class LangNet(nn.Module):
    def __init__(self, configs):
        super(LangNet, self).__init__()
        self.text_dims = configs['text_dims']
        self.use_view_hierarchy = (configs['use_view_text'] or configs['use_view_ids'] or configs['use_view_cls'])
        self.use_view_text = configs['use_view_text']
        self.use_view_ids = configs['use_view_ids']
        self.use_view_cls = configs['use_view_cls']
        self.max_views = configs['views']
        self.view_join = configs['join_type']
        self.view_text_join = configs['view_text_join']
        self.view_add_on_dims = 0

        if self.use_view_text:
            self.view_add_on_dims += configs['view_text_count']
        if self.use_view_ids:
            self.view_add_on_dims += configs['view_ids_count']
        if self.use_view_cls:
            self.view_add_on_dims += configs['view_cls_count']

        self.vocab_size = configs['vocab_size']
        self.embed = nn.Embedding(self.vocab_size, self.text_dims)
        self.embed.load_state_dict({'weight': torch.from_numpy(np.load(os.path.join(configs['data_dir'], configs['vocab_path'])))})

        # how to join view hierarchy text / ids / class tokens within a view
        if self.view_text_join == 'AVG':
            self.view_text_dims = self.text_dims
        else:
            # CONCAT
            self.view_text_dims = self.view_add_on_dims * self.text_dims

        # how to join view hierarchy text / ids / class tokens across view
        if self.view_join == 'LSTM':
            self.lstm = nn.LSTM(self.view_text_dims, 512, batch_first=True)
            self.vh_add_on_dims = 512
        elif self.view_join == 'AVG':
            self.vh_add_on_dims = self.view_text_dims
        else:
            # CONCAT
            self.vh_add_on_dims = self.max_views * self.view_text_dims

        # assuming average embedding of text command tokens
        if self.use_view_hierarchy:
            self.fc1 = nn.Linear(self.text_dims + self.vh_add_on_dims, 512)
        else:
            self.fc1 = nn.Linear(self.text_dims, 512)

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_task, x_view_text):
        task_embed = self.embed(x_task)
        task_embed = torch.mean(task_embed, dim=1)
        view_text_embed = self.embed(x_view_text)

        if self.view_text_join == 'AVG':
            view_text_embed = torch.mean(view_text_embed, dim=2)
        else:
            # CONCAT
            view_text_embed = torch.flatten(view_text_embed, start_dim=2)

        if self.view_join == 'LSTM':
            lstm_output, (hidden_state, cell_state) = self.lstm(view_text_embed)
            view_text_view_embed = torch.squeeze(hidden_state)
        elif self.view_join == 'AVG':
            view_text_view_embed = torch.squeeze(torch.mean(view_text_embed, dim=1))
        else:
            # CONCAT
            view_text_view_embed = torch.flatten(view_text_embed, start_dim=1)

        if self.use_view_hierarchy:
            input_feats = normalize(torch.cat([task_embed, view_text_view_embed], dim=1))
        else:
            input_feats = normalize(task_embed)

        hidden = self.relu(self.fc1(input_feats))
        hidden2 = self.relu(self.fc2(hidden))
        output = self.sigmoid(self.fc3(hidden2))

        return output


class ComboNet(nn.Module):
    def __init__(self, configs):
        super(ComboNet, self).__init__()
        self.text_dims = configs['text_dims']
        self.max_views = configs['views']
        self.join = configs['join_type']
        self.crop_join = configs['crop_join']
        self.num_crops = configs['num_crops']
        self.feats_1 = list(configs['feat_dims'].keys())[0]
        self.feats_2 = list(configs['feat_dims'].keys())[1]
        self.view_text_join = configs['view_text_join']
        assert len(list(configs['feat_dims'].keys())) == 2
        self.vocab_size = configs['vocab_size']
        self.embed = nn.Embedding(self.vocab_size, self.text_dims)
        self.embed.load_state_dict({'weight': torch.from_numpy(np.load(os.path.join(configs['data_dir'], configs['vocab_path'])))})
        
        self.input_feat_dims = sum(list(configs['feat_dims'].values()))

        if self.join == 'LSTM':
            self.lstm = nn.LSTM(self.input_feat_dims, 512, batch_first=True)
            self.output_feat_dims = 512
        elif self.join == 'AVG':
            self.output_feat_dims = self.input_feat_dims
        else:
            # CONCAT
            self.output_feat_dims = self.max_views * self.input_feat_dims

        self.fc1 = nn.Linear(self.text_dims + self.output_feat_dims, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_text, x_feat_1, x_feat_2):
        # embed text features
        text_embed = torch.mean(self.embed(x_text), dim=1)
        if self.feats_1 in ['RESNET', 'ICON']:
            # average over 10 crops or 57 icons
            x_feat_1 = torch.mean(x_feat_1, dim=2)
        if self.feats_2 in ['RESNET', 'ICON']:
            # vis_feats_2 = resnet or icon
            x_feat_2 = torch.mean(x_feat_2, dim=2)

        if self.feats_1 in ['et', 'et_id', 'et_id_cls']:
            view_text_embed = self.embed(x_feat_1)
            if self.view_text_join == 'AVG':
                x_feat_1 = torch.mean(view_text_embed, dim=2)
            else:
                # CONCAT
                x_feat_1 = torch.flatten(view_text_embed, start_dim=2)
        if self.feats_2 in ['et', 'et_id', 'et_id_cls']:
            view_text_embed = self.embed(x_feat_2)
            if self.view_text_join == 'AVG':
                x_feat_2 = torch.mean(view_text_embed, dim=2)
            else:
                # CONCAT
                x_feat_2 = torch.flatten(view_text_embed, start_dim=2)

        feats = torch.cat([x_feat_1, x_feat_2], dim=2)
        if self.join == 'LSTM':
            lstm_output, (hidden_state, cell_state) = self.lstm(feats)
            feat_embed = torch.squeeze(hidden_state)
        elif self.join == 'AVG':
            feat_embed = torch.squeeze(torch.mean(feats, dim=1))
        else:
            # concat
            feat_embed = torch.flatten(feats, start_dim=1)

        input_feats = normalize(torch.cat([text_embed, feat_embed], dim=1))

        hidden = self.relu(self.fc1(input_feats))
        hidden2 = self.relu(self.fc2(hidden))
        output = self.sigmoid(self.fc3(hidden2))

        return output