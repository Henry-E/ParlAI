import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import ipdb

# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------
RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.rnn_type = RNN_TYPES[opt['rnn_type']]
        self.input_size = opt['embedding_dim']   # TODO this may not be available
        self.hidden_size = opt['embedding_dim']   # TODO change both of these away from embedding dim

        self.rnn = self.rnn_type(self.input_size, self.hidden_size)

    def forward(self, input):
        outputs, hidden_init = self.rnn(input)
        return outputs, hidden_init

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.rnn_type = RNN_TYPES[opt['rnn_type']]
        self.input_size = opt['embedding_dim']   # TODO this may not be available
        self.hidden_size = opt['embedding_dim']   # TODO change both of these away from embedding dim

        self.rnn = self.rnn_type(self.input_size, self.hidden_size)

    def forward(self, input, hidden_init):
        # Note that the linear transformation and linear softmax happens in a 
        # separate memory efficient loss function 
        outputs, _ = self.rnn(input, hidden_init)
        return outputs

        