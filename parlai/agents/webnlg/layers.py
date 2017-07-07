import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import ipdb

RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.rnn_type = RNN_TYPES[opt['rnn_type']]
        self.input_size = opt['embedding_dim']
        self.hidden_size = opt['hidden_size']

        self.rnn = self.rnn_type(self.input_size, self.hidden_size)

    def forward(self, input):
        outputs, hidden_init = self.rnn(input)
        return outputs, hidden_init

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.rnn_type = RNN_TYPES[opt['rnn_type']]
        self.input_size = opt['embedding_dim']
        self.hidden_size = opt['hidden_size']

        self.rnn = self.rnn_type(self.input_size, self.hidden_size)
        self.attn = Attention(opt)

    def forward(self, input, hidden, context):
        # Note that the linear transformation and softmax happen in a separate
        # memory efficient loss function
        outputs = []
        for i, input_t in enumerate(input.split(1)):
            rnn_output, hidden = self.rnn(input_t, hidden)
            attn_output, attn = self.attn(rnn_output.squeeze(),
                                            context.transpose(0, 1))
            output = attn_output    # TODO start using dropout
            outputs += [output]
        outputs = torch.stack(outputs)
        # TODO store, stack and return attentions
        return outputs

class Attention(nn.Module):
    # For the time being we're implementing Luong attention
    # https://arxiv.org/pdf/1508.04025.pdf
    def __init__(self, opt):
        super(Attention, self).__init__()

        self.hidden_size = opt['hidden_size']
        self.linear_in = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_out = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()

        # TODO implement coverage

        # TODO possibly implement bahdanau attention

    def forward(self, input, context):
        # based on OpenNMT's GlobalAttention
        # https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/GlobalAttention.py
        targetT = self.linear_in(input).unsqueeze(2)

        attn = torch.bmm(context, targetT).squeeze(2)
        attn = self.sm(attn)

        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weightedContext = torch.bmm(attn3, context).squeeze(1)
        contextCombined = torch.cat((weightedContext, input), 1)

        final = self.linear_out(contextCombined)
        contextOutput = self.tanh(final)

        return contextOutput, attn







