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
        
        triples_emb = self.embedding(triples)
        text_emb = self.embedding(text)

        outputs, hidden_init = self.encoder(triples_emb)
        outputs = self.decoder(text_emb, hidden_init)

        return outputs

    # def generate():
        # TODO returns a list of words indices instead of loss