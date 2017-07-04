import torch
import torch.nn as nn
from . import layers


from torch.autograd import Variable   # consider deleting and doing earlier
import ipdb

class RnnTriples2Text(nn.Module):
    '''Network for reading triples and returning text for WebNLG challenge'''

    def __init__(self, opt, padding_idx=0):
        super(RnnTriples2Text, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(opt['vocab_size'],
                                      opt['embedding_dim'],
                                      padding_idx=padding_idx)

        # simple RNN encoder
        self.encoder = layers.Encoder(opt)

        # simple RNN decoder
        self.decoder = layers.Decoder(opt)

    def forward(self, triples, text):
        ''' TODO list inputs
        '''

        ipdb.set_trace()
        # TODO When not testing we will be applying Variable earlier in process
        triples_emb = self.embedding(Variable(triples))
        text_emb = self.embedding(Variable(text))

        outputs, hidden_init = self.encoder(triples_emb)

        # TODO decoder call and forward pass code for decoder

    # def generate():
        # TODO returns a list of words indices instead of loss