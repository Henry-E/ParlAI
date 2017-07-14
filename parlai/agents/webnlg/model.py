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
        padding_idx = 0
        self.criterion = nn.NLLLoss(ignore_index=padding_idx)

        # Building probability generator as part of memory efficient loss
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
        self.generator.train()

        # Clear gradients
        self.optimizer.zero_grad()

        # We don't expand out the batch contents in this fuction, is this 
        # potentially confusing? Also is cloning going to mess with the
        # gradients? Probably they get propagated through outputs' variable
        # TODO use a cleaner method to pass cloned variables to the loss
        triples = batch[0].clone()
        targets = batch[1].clone()

        # Run forward
        outputs, attentions, p_gens = self.network(*batch)

        # Calculate loss and run backward
        loss = self._memoryEfficientLoss(outputs, attentions, p_gens, triples, 
                                        targets, self.generator, self.criterion)
        self.train_loss.update(loss.data[0], targets.size(1))     # TODO check n is correct

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

    def _memoryEfficientLoss(self, outputs, attentions, p_gens, triples, 
            targets, generator, criterion, evaluate=False, vocab_size=7000):
        # TODO in the future maybe we will make this more memory efficient by
        # cutting off the link back to earlier variables and updating grad
        # outside of this function like is done in OpenNMT. The fiddlyness
        # is having multiple variables that require grad updating
        if evaluate:
            outputs = Variable(outputs.data, requires_grad=(not evaluate), volatile=evaluate)
            attentions = Variable(attentions.data, requires_grad=(not evaluate), volatile=evaluate)
            p_gens = Variable(p_gens.data, requires_grad=(not evaluate), volatile=evaluate)
        
        batch_size = outputs.size(1)
        target_seq_len = outputs.size(0)
        triples_seq_len = triples.size(0)
        max_dict_index = triples.max().data[0]+1

        # Here we are assigning the probabilities for each vocab index that
        # appears in the encoder from the decoder attentions. This is using the
        # extended vocabulary indices & is what allows us to copy unknown words
        attention_scores = Variable(torch.zeros(target_seq_len, 
                                                batch_size,
                                                max_dict_index))
        if self.opt['cuda']:
            attention_scores = attention_scores.cuda()
        triples_reshaped = triples.t().expand(target_seq_len,
                                              batch_size,
                                              triples_seq_len)
        attention_scores.scatter_add(2, triples_reshaped, attentions)
        attention_scores = attention_scores.view(-1, max_dict_index)

        # TODO set a maximum sequence length to process the batches by, and 
        # split the sequence into batches of that length if memory usage is too
        # high

        # note that by not renaming variables requires_grad breaks seemingly
        outputs = outputs.view(-1, outputs.size(2))
        scores = generator(outputs)
        extra_zeros = Variable(torch.zeros(scores.size(0), max_dict_index - scores.size(1)))
        if self.opt['cuda']:
            extra_zeros =  extra_zeros.cuda()
        scores = torch.cat((scores, extra_zeros), 1)

        # a bit of jiggerypokery to get p_gens from size 1 X batch_size X 1 to
        # size (target_seq_len * batch_size) X 1 so that it can be broadcast 
        # against the scores
        p_gens = p_gens.squeeze().expand(target_seq_len, batch_size).contiguous().view(-1).unsqueeze(1)

        final_scores = p_gens * scores + (1 - p_gens) * attention_scores
        # print(targets.max().data[0], final_scores.size())
        loss = criterion(final_scores, targets.view(-1))

        if not evaluate:
            loss.backward()

        # In OpenNMT's implementation they only backprop as far the start of 
        # this function and then they return the grads for each variable
        # which are then passed back through the full graph.
        # grad_output = None if outputs.grad is None else outputs.grad.data
        return loss

    def cuda(self):
        self.network.cuda()
        self.generator.cuda()
        self.criterion.cuda()



    
