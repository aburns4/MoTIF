import torch
import torch.nn as nn
from Screen2Vec import Screen2Vec

# contains models that handle the step of predicting the next screen in the trace
# these are the model parts that are trained, but not part of generating the actual
# screen vectors for individual screens

class TracePredictor(nn.Module):
    """
    predicts the embeddings of the next screen in a trace based on its preceding screens
    """
    def __init__(self, embedding_model: Screen2Vec, net_version: int):
        super().__init__()
        self.model = embedding_model
        self.bert_size = self.model.bert_size
        self.net_version = net_version
        self.combiner = nn.LSTM(self.bert_size, self.bert_size, batch_first=True)

    def forward(self, UIs, descr, trace_screen_lengths, layouts=None, cuda=True):
        """
        UIs:    embeddings of all UI elements on each screen, padded to the same length
                batch_size x screen_size x trace_length x bert_size + additional_ui_size
        descr:  Sentence BERT embeddings of app descriptions
                batch_size x trace_length x bert_size
        trace_screen_lengths: length of UIs before zero padding was performed
                batch_size x trace_length
        layouts: (None if not used in this net version) the autoencoded layout vector for the screen
                batch_size x trace_length x additonal_size_screen
        cuda:   True if TracePredictor has been sent to GPU, False if not
        """
        # embed all of the screens using Screen2Vec
        screens = self.model(UIs, descr, trace_screen_lengths, layouts)

        # take all but last element of each trace, store as context
        # last element is the desired result/target
        if cuda:
            context = torch.narrow(screens, 1, 0, screens.size()[1]-1).cuda()
            result = torch.narrow(screens, 1, screens.size()[1]-1, 1).squeeze(1).cuda()
        else:
            context = torch.narrow(screens, 1, 0, screens.size()[1]-1)
            result = torch.narrow(screens, 1, screens.size()[1]-1, 1).squeeze(1)
        if self.net_version == 9:
            # baseline option
            h = torch.sum(context, dim = -2)/len(context)
            return h, result, context
        # run screens in trace through model to predict last one
        output, (h,c) = self.combiner(context)
        if self.net_version == 5:
            descriptions = torch.narrow(descr, 1, 0, 1)
            h = torch.cat((h[0], descriptions.squeeze(1)), dim=-1)
            result = torch.cat((result,descriptions.squeeze(1)), dim=-1)
            context = torch.cat((context, torch.narrow(descr,1,0,descr.size()[1]-1).squeeze(1)), dim=-1)
        else:
            h = h[0]
        return h, result, context

class BaselinePredictor(nn.Module):
    """
    predicts the embeddings of the next screen in a trace based on its preceding screens
    using baseline model embeddings of the screen
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.emb_size = embedding_size
        self.combiner = nn.LSTM(self.emb_size, self.emb_size, batch_first=True)

    def forward(self, embeddings, cuda=True):
        # embed all of the screens using Screen2Vec
        
        # run screens in trace through model to predict last one
        output, (h,c) = self.combiner(embeddings)
        h = h[0]
        return h