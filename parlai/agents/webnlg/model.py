import torch
# TODO possibly outsource optim to the OpenNMT module or one of the
# intelligent update optimisers we found before on github
import torch.optim as optim
import torch.nn.functional as functional
import logging

from torch.autograd import Variable
# TODO utility function for load pre-trained embeddings, AverageMeter
from .rnn_triples2text import RnnTriples2Text

logger = logging.getLogger('WebNLG')

class Triples2TextModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, word_dict, state_dict=None):
        # Book-keeping
        self.opt = opt
        self.word_dict = word_dict
        self.updates = 0
        # TODO self.train_loss = AverageMeter(), or maybe we will use 
        # tensorboard logger instead

        # Building network
        self.network = RnnTriples2Text(opt)
        # if state_dict:
        # TODO loading a presaved model

        # Building optimizer
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

    # def set_embeddings(self):
        # TODO something about customising the use of embeddings

    def update(self, batch):
        # Train mode
        self.network.train()

        # TODO Set to GPU, this might not be the best place to set to GPU?

        # Run forward
        outputs = self.network(*batch)

        # TODO Compute loss

        # TODO Clear gradients and run backward

        # TODO Clip graidents, though we might be able to do this in an 
        # optimiser function

        # Update parameters

    # def evaluate(self, batch):
        # TODO

    # def generate(self, batch):
        # TODO

    # def save(self, filename):
        # TODO
