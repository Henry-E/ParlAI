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
        self.pgen = GenerationProbability(opt)

    def forward(self, input, hidden, context):
        # Note that the linear transformation and softmax happen in a separate
        # memory efficient loss function
        outputs = []
        attentions = []
        p_gens = []
        for i, input_t in enumerate(input.split(1)):
            rnn_output, hidden = self.rnn(input_t, hidden)
            attn_output, attn = self.attn(rnn_output.squeeze(),
                                            context.transpose(0, 1))
            output = attn_output    # TODO start using dropout
            # TODO make p_gen optional and add option to config file
            p_gen = self.pgen(attn, context.transpose(0, 1), rnn_output, input_t)
            outputs += [output]
            attentions += [attn]
            p_gens = [p_gen]
        outputs = torch.stack(outputs)
        attentions = torch.stack(attentions)
        p_gens = torch.stack(p_gens)
        return outputs, attentions, p_gens

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

    def forward(self, decoder_state, context):
        # based on OpenNMT's GlobalAttention
        # https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/GlobalAttention.py
        targetT = self.linear_in(decoder_state).unsqueeze(2)

        attn = torch.bmm(context, targetT).squeeze(2)
        attn = self.sm(attn)

        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weightedContext = torch.bmm(attn3, context).squeeze(1)
        contextCombined = torch.cat((weightedContext, decoder_state), 1)

        final = self.linear_out(contextCombined)
        attn_output = self.tanh(final)
        return attn_output, attn

class GenerationProbability(nn.Module):
    # Following equation 8 in the pointer networks summary paper
    def __init__(self, opt):
        super(GenerationProbability, self).__init__()

        self.hidden_size = opt['hidden_size']
        self.embedding_dim = opt['embedding_dim']
        # TODO decide whether or not to remove bias
        self.linear_pgen = nn.Linear((2*self.hidden_size+self.embedding_dim), 1)
        self.sig = nn.Sigmoid()

    def forward(self, attn, context, decoder_state, decoder_input):
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        
        inputs = torch.cat((weighted_context, 
            decoder_state.squeeze(), decoder_input.squeeze()), 1)
        outputs = self.linear_pgen(inputs)
        p_gen = self.sig(outputs)
        return p_gen







