import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import unicodedata
from collections import Counter
import ipdb

# ------------------------------------------------------------------------------
# General logging utilities.
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ----------------------------------------------------------------------------
# Data/model utilities.
# ----------------------------------------------------------------------------

def normalize_text(text):
        return unicodedata.normalize('NFD', text)

def load_embeddings(opt, word_dict, padding_idx=0):
    """Initialize embeddings from file of pretrained vectors."""
    embeddings = nn.Embedding(opt['vocab_size'],
                              opt['embedding_dim'],
                              padding_idx=padding_idx)

    # Fill in embeddings
    if not opt.get('embedding_file'):
        raise RuntimeError('Tried to load embeddings with no embedding file.')
    with open(opt['embedding_file']) as f:
        for line in f:
            parsed = line.rstrip().split(' ')
            assert(len(parsed) == opt['embedding_dim'] + 1)
            w = normalize_text(parsed[0])
            if w in word_dict:
                vec = torch.Tensor([float(i) for i in parsed[1:]])
                # I remember reading about inplace operations being bad for
                # variables. Is it simply cheating by assigning it to [].data?
                embeddings.weight[word_dict[w]].data = vec             
    return embeddings