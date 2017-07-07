import torch
# TODO possibly outsource optim to the OpenNMT module or one of the
# intelligent update optimizers we found before on github
import torch.optim as optim
import torch.nn as nn
import logging

from torch.autograd import Variable
from .utils import AverageMeter
from .rnn_triples2text import RnnTriples2Text

import ipdb

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
        self.train_loss = AverageMeter()

        # Building network
        self.network = RnnTriples2Text(opt)
        # if state_dict:
        # TODO loading a presaved model

        # Building criterion / loss function
        self.criterion = self._criterion(self.opt['vocab_size'])

        # Building probability generator as part of memory efficient loss
        # TODO it's possible vocab size might be variable in pointer network
        self.generator = nn.Sequential(
            nn.Linear(self.opt['hidden_size'], self.opt['vocab_size']),
            nn.LogSoftmax())

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

        # Clear gradients
        self.optimizer.zero_grad()

        # Run forward
        outputs = self.network(*batch)

        # We don't expand out the batch contents in this fuction, is this 
        # potentially confusing?
        target = batch[1]
        # Calculate loss and run backward
        loss = self._memoryEfficientLoss(outputs, target, self.generator, self.criterion)
        self.train_loss.update(loss.data[0], target.size(1))     # TODO check n is correct

        # TODO Clip graidents, though we might be able to do this in an 
        # optimizer function

        # Update parameters
        self.optimizer.step()
        self.updates += 1

    # def evaluate(self, batch):
        # TODO

    # def generate(self, batch):
        # TODO

    # def save(self, filename):
        # TODO

    # ------------------------------------------------------------------------
    # Model helper functions
    # ------------------------------------------------------------------------

    def _criterion(self, vocab_size, padding_idx=0):
        weight = torch.ones(vocab_size)
        weight[padding_idx] = 0
        criterion = nn.NLLLoss(weight, size_average=False)
        return criterion

    def _memoryEfficientLoss(self, outputs, targets, generator, criterion, evaluate=False):
        outputs = Variable(outputs.data, requires_grad=(not evaluate), volatile=evaluate)

        batch_size = outputs.size(1)
        # TODO set a maximum sequence length to process the batches by, and 
        # split the sequence into batches of that length if memory usage is too
        # damn high
        outputs = outputs.view(-1, outputs.size(2))
        scores = generator(outputs)
        loss = criterion(scores, targets.view(-1))
        if not evaluate:
            # TODO figure out why they divide loss by batch size but not by
            # number of non-padding words as well? Could be that it's 
            # unecessary but I'd like to understand why better
            loss.div(batch_size).backward()

        # TODO Figure out if we need to return output.grad as well or not

        return loss

    def cuda(self):
        self.network.cuda()
        self.generator.cuda()
        self.criterion.cuda()



    
