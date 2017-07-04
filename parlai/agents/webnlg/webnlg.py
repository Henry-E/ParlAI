# a first attempt at implementing a custom agent class 

import torch
import os
import numpy as np
import logging
import copy
import ipdb

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from .model import Triples2TextModel

class WebnlgAgent(Agent):
    # TODO staticmethod for parsing command line arguments from a config file
    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)


    def __init__(self, opt, shared=None):
        if opt['numthreads'] >1:
            raise RuntimeError("numthreads > 1 not supported for this model.")

        # Load dictionary
        if not shared:
            word_dict = DictionaryAgent(opt)
        # TODO I don't think we need to keep track of episodes

        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
        self.id = self.__class__.__name__
        self.word_dict = word_dict
        self.opt = copy.deepcopy(opt)
        # TODO config.set_defaults(self.opt)

        self._init_from_scratch()
        # TODO initialise model from saved 

        # TODO set model to GPU
        self.n_examples = 0

    def _init_from_scratch(self):
        # TODO options for initialising the model
        self.opt['vocab_size'] = len(self.word_dict)
        # TODO move all these default options over to a config.py file
        self.opt['embedding_dim'] = 300    
        self.opt['rnn_type'] = 'gru'   
        self.opt['optimizer'] = 'sgd' 
        self.opt['learning_rate'] = 0.1
        self.opt['momentum'] = 0
        self.opt['weight_decay'] = 0

        print('[ Initializing model from scratch ]')
        self.model = Triples2TextModel(self.opt, self.word_dict)
        # TODO set embeddings

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        self.observation = observation
        return observation

    def act(self):
        """Update or predict on a single example (batchsize = 1)."""
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")
        reply = {'id': self.getID()}
        example = self._build_example(self.observation)
        if example is None:
            return reply
        # for a batch size of one all we need to do is add an extra dimension
        batch = self.batchify(example)

        # TODO train
        self.n_examples += 1 
        self.model.update(batch)

        # TODO predict

        return reply

    # ------------------------------------------------------------------------
    # Helper functions.
    # ------------------------------------------------------------------------

    def _build_example(self, example):
        triples = self.word_dict.txt2vec(example['text'])
        target = self.word_dict.txt2vec(example['labels'])
        triples = torch.LongTensor(triples)
        target = torch.LongTensor(target)
        return triples, target

    # TODO move batchify to util.py module
    def batchify(self, batch, null=0, cuda=False):
        # TODO it's kind of hard to program this bit for multiple examples
        # without really understanding what data structures we're dealing
        # with

        # TODO only works for batch size of 1, scale up when possible
        triples = batch[0].unsqueeze(1)
        target = batch[1].unsqueeze(1)

        # # Batch triples
        # max_length = max([t.size(0) for t in triples])



        # # Batch targets
        # max_length = max([t.size(0) for t in targets])
        return triples, target

    def report(self):
        return (
            '[train] updates = %d | train loss = %.2f | exs = %d' %
            (self.model.updates, self.model.train_loss.avg, self.n_examples)
            )








