import torch
# TODO possibly outsource optim to the OpenNMT module or one of the
# intelligent update optimizers we found before on github
import torch.optim as optim
import torch.nn as nn
import logging

from torch.autograd import Variable
from .utils import load_embeddings, AverageMeter
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
        self.valid_loss = AverageMeter()

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

    def set_embeddings(self):
        # Read word embeddings.
        if not self.opt.get('embedding_file'):
            logger.warning('[ WARNING: No embeddings provided. '
                           'Keeping random initialization. ]')
            return
        logger.info('[ Loading pre-trained embeddings ]')
        embeddings = load_embeddings(self.opt, self.word_dict)
        logger.info('[ Num embeddings = %d ]' % embeddings.size(0))

        # Sanity check dimensions
        new_size = embeddings.size()
        old_size = self.network.embedding.weight.size()
        if new_size[1] != old_size[1]:
            raise RuntimeError('Embedding dimensions do not match.')
        if new_size[0] != old_size[0]:
            logger.warning(
                '[ WARNING: Number of embeddings changed (%d->%d) ]' %
                (old_size[0], new_size[0])
            )

        # Swap weights
        self.network.embedding.weight.data = embeddings

        # # If partially tuning the embeddings, keep the old values
        # if self.opt['tune_partial'] > 0:
        #     if self.opt['tune_partial'] + 2 < embeddings.size(0):
        #         fixed_embedding = embeddings[self.opt['tune_partial'] + 2:]
        #         self.network.fixed_embedding = fixed_embedding

    def update(self, batch):
        # Train mode
        self.network.train()
        self.generator.train()

        # Clear gradients
        self.optimizer.zero_grad()

        # TODO use a cleaner method to pass cloned variables to the loss.
        # maybe just clone the whole batch and pass it to the loss function?
        triples = batch[0].clone()
        targets = batch[1].clone()

        # Run forward
        outputs, attentions, p_gens = self.network(*batch)

        # Calculate loss and run backward
        loss, _ = self._memoryEfficientLoss(outputs, attentions, p_gens, triples, 
                                        targets, self.generator, self.criterion)
        # TODO check number of targets is correct and is ignoring 
        # padding indices
        self.train_loss.update(loss.data[0], targets.size(1))    

        # TODO Clip gradients, though we might be able to do this in an 
        # optimizer function

        # Update parameters
        self.optimizer.step()
        self.updates += 1

    def evaluate(self, batch):
        # Eval mode
        self.network.eval()
        self.generator.eval()

        triples = batch[0].clone()
        targets = batch[1].clone()

        # Run forward
        outputs, attentions, p_gens = self.network(*batch)

        loss, _ = self._memoryEfficientLoss(outputs, attentions, p_gens, triples, 
                                targets, self.generator, self.criterion,
                                evaluate=True)
        self.valid_loss.update(loss.data[0], targets.size(1))


    def generate(self, batch, start_idx=3, end_idx=1):
        # Eval mode
        self.network.eval()
        self.generator.eval()

        triples = batch[0]
        triples_encode = batch[0].clone()
        context, hidden_init = self.network.encode_triples(triples_encode)
        hidden = hidden_init
        # for now assume batch size could be greater than one
        init_token = Variable(torch.LongTensor([start_idx]).expand(1, triples.size(1)))
        # TODO remove fake targets vector and make probability calculation 
        # its own function. The target vector has 1D shape Batch Size
        targets = Variable(torch.LongTensor([start_idx]).expand(1, triples.size(1)))
        if self.opt['cuda']:
            init_token = init_token.cuda()
            targets = targets.cuda()
        token_embedding = self.network.embedding(init_token)
        sentence = []
        while len(sentence) < 40:
            outputs, attentions, p_gens = self.network.decoder(token_embedding, hidden, context)
            # We're being kind of lazy for the moment by getting the loss function
            # to just return the final word probabilities rather than making the 
            # probability calculation into its own function
            _, final_scores = self._memoryEfficientLoss(outputs, attentions, p_gens, triples, 
                        targets, self.generator, self.criterion,
                        evaluate=True)
            topv, topi = final_scores.topk(1)
            if topi.data[0][0] is end_idx:
                break
            sentence += [topi.data[0][0]]
            hidden = outputs
            next_token = self.network.replace_extended_vocabulary(topi.t().clone())
            token_embedding = self.network.embedding(next_token)
        return sentence

    # def save(self, filename):
        # TODO

    # ------------------------------------------------------------------------
    # Model helper functions
    # ------------------------------------------------------------------------

    def _memoryEfficientLoss(self, outputs, attentions, p_gens, triples, 
            targets, generator, criterion, evaluate=False, pointer_network=False):
        # TODO in the future maybe we will make this more memory efficient by
        # cutting off the link back to earlier variables and updating grad
        # outside of this function like is done in OpenNMT. The fiddlyness
        # is having multiple variables that require grad updating
        if evaluate:
            outputs = Variable(outputs.data, requires_grad=(not evaluate), volatile=evaluate)
            attentions = Variable(attentions.data, requires_grad=(not evaluate), volatile=evaluate)
            p_gens = Variable(p_gens.data, requires_grad=(not evaluate), volatile=evaluate)
        
        if pointer_network:
            # TODO The maths behind how we're calculating the loss is wrong 
            # and needs a little bit of thought to get working. We still 
            # have to check how the abstract summarization paper deals with
            # this.
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
            # We reshape the input sequence to match the size of the attentions
            triples_reshaped = triples.t().expand(target_seq_len,
                                                  batch_size,
                                                  triples_seq_len)
            attention_scores.scatter_add(2, triples_reshaped, attentions)
            attention_scores = attention_scores.view(-1, max_dict_index)
            if max_dict_index < self.opt['vocab_size']:
                extra_zeros = Variable(torch.zeros(batch_size*target_seq_len, self.opt['vocab_size'] - max_dict_index))
                if self.opt['cuda']:
                    extra_zeros =  extra_zeros.cuda()
                attention_scores = torch.cat((attention_scores, extra_zeros), 1)

            # TODO set a maximum sequence length to process the batches by, and 
            # split the sequence into batches of that length if memory usage is too
            # high

            # note that not renaming variables seemingly breaks requires_grad
            outputs = outputs.view(-1, outputs.size(2))
            scores = generator(outputs)
            if max_dict_index > self.opt['vocab_size']:
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
        else:
            outputs = outputs.view(-1, outputs.size(2))
            final_scores = generator(outputs)
            self.network.replace_extended_vocabulary(targets)   
            loss = criterion(final_scores, targets.view(-1))


        if not evaluate:
            loss.backward()

        # In OpenNMT's implementation they only backprop as far the start of 
        # this function and then they return the grads for each variable
        # which are then passed back through the full graph.
        # grad_output = None if outputs.grad is None else outputs.grad.data
        return loss, final_scores

    def cuda(self):
        self.network.cuda()
        self.generator.cuda()
        self.criterion.cuda()



    
