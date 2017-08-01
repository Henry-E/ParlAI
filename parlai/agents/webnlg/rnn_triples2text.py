import torch
import torch.nn as nn
from . import layers


from torch.autograd import Variable   # consider deleting and doing earlier

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
        context, hidden_init = self.encode_triples(triples)

        text = self.replace_extended_vocabulary(text)
        text_emb = self.embedding(text)
        outputs, attentions, p_gens = self.decoder(text_emb, hidden_init, context)

        return outputs, attentions, p_gens

    def replace_extended_vocabulary(self, indices, unk_idx=2):
        # Because we extend the vocabulary for the pointer network we need to
        # replace all the indices greater than the original vocab size with
        # the unknown token in order for the model to work as normal
        indices[indices>self.opt['vocab_size']-1] = unk_idx
        return indices

    def encode_triples(self, triples):
        triples = self.replace_extended_vocabulary(triples)
        triples_emb = self.embedding(triples)
        context, hidden_init = self.encoder(triples_emb)
        return context, hidden_init



    # def generate():
        # TODO returns a list of words indices instead of loss