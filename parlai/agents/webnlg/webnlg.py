# a first attempt at implementing a custom agent class 

import torch
import os
import numpy as np
import logging
import copy
import ipdb

from torch.autograd import Variable
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from . import config
from .model import Triples2TextModel

# ----------------------------------------------------------------------------
# Dictionary.
# ----------------------------------------------------------------------------

class SimpleDictionaryAgent(DictionaryAgent):
    # custom dictionary agent to only use pre-trained embeddings and extend
    # dictionary for the pointer network

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--pretrained_words', type='bool', default=True,
            help='Use only words found in provided embedding_file'
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Index words in embedding file
        if self.opt['pretrained_words'] and self.opt.get('embedding_file'):
            print('[ Indexing words with embeddings... ]')
            self.embedding_words = set()
            with open(self.opt['embedding_file']) as f:
                for line in f:
                    w = normalize_text(line.rstrip().split(' ')[0])
                    self.embedding_words.add(w)
            print('[ Num words in set = %d ]' %
                  len(self.embedding_words))
        else:
            self.embedding_words = None

    def add_to_dict(self, tokens):
        """Builds dictionary from the list of provided tokens.
        Only adds words contained in self.embedding_words, if not None.
        """
        for token in tokens:
            if (self.embedding_words is not None and
                token not in self.embedding_words):
                continue
            self.freq[token] += 1
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

    def extend_dict(self, text):
        """ Add words that appear in the input to the dictionary for possible use
        later on in the pointer network copying mechanism
        """
        tokens = self.tokenize(text)

        for token in tokens:
            # We don't really care about frequency of UNKs
            if token not in self.tok2ind:
                index = len(self.tok2ind)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

class WebnlgAgent(Agent):
    # TODO staticmethod for parsing command line arguments from a config file
    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        WebnlgAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return SimpleDictionaryAgent


    def __init__(self, opt, shared=None):
        if opt['numthreads'] >1:
            raise RuntimeError("numthreads > 1 not supported for this model.")

        # Load dictionary
        if not shared:
            word_dict = WebnlgAgent.dictionary_class()(opt)
        # TODO I don't think we need to keep track of episodes

        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
        self.id = self.__class__.__name__
        self.word_dict = word_dict
        self.opt = copy.deepcopy(opt)
        config.set_defaults(self.opt)

        self._init_from_scratch()

        # TODO option to initialise model from saved 

        self.opt['cuda'] = not self.opt['no_cuda'] and torch.cuda.is_available()
        if self.opt['cuda']:
            print('[ Using CUDA (GPU %d) ]' % opt['gpu'])
            torch.cuda.set_device(opt['gpu'])
            self.model.cuda()
        self.n_examples = 0

    def _init_from_scratch(self):
        self.opt['vocab_size'] = len(self.word_dict)

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
        batch = self._batchify([example])

        # Either train and/or generate
        self.n_examples += 1 
        self.model.update(batch)
        # TODO generate

        return reply

    def batch_act(self, observations):
        """Update or predict on a batch of examples.
        More efficient than act()
        """
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported")

        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        examples = [self._build_example(obs) for obs in observations]

        batch = self._batchify(examples)

        # Either train and/or generate
        self.n_examples += len(examples)
        self.model.update(batch)
        # TODO generate

        return batch_reply




    # ------------------------------------------------------------------------
    # Helper functions.
    # ------------------------------------------------------------------------

    def _build_example(self, example):
        # Add words which appear in input but not in dictionary to dictionary
        # to be used with pointer network later on
        # TODO for efficiency make this only happen during the first epoch
        # TODO possibly make this optional? It doesn't really affect anything
        # by having it as default though
        self.word_dict.extend_dict(example['text'])

        triples = self.word_dict.txt2vec(example['text'])
        target = self.word_dict.txt2vec(example['labels'])
        triples = torch.LongTensor(triples)
        target = torch.LongTensor(target)
        return triples, target

    def _batchify(self, batch, padding_idx=0):
        # TODO it's kind of hard to program this bit for multiple examples
        # without really understanding what data structures we're dealing
        # with

        # TODO the sequence of action is the same for both so we could probably
        # put this into its own function
        triples = [ex[0] for ex in batch]
        target = [ex[1] for ex in batch]

        # Batch triples
        # a very primitive form of padding
        max_length = max([t.size(0) for t in triples])
        # we're using shape SeqLen X BatchLen which is the reverse of what's
        # expected in the embedding function & drqa but correct order for GRU 
        triples_batch = torch.LongTensor(max_length, len(batch)).fill_(padding_idx)
        for col, triple in enumerate(triples):
            triples_batch[0:len(triple), col] = triple

        max_length = max([t.size(0) for t in target])
        target_batch = torch.LongTensor(max_length, len(batch)).fill_(padding_idx)
        for col, targ in enumerate(target):
            target_batch[0:len(targ), col] = targ

        triples_variable = Variable(triples_batch)
        target_variable = Variable(target_batch)

        if self.opt['cuda']:
            triples_variable = triples_variable.cuda()
            target_variable = target_variable.cuda()

        return triples_variable, target_variable

    def report(self):
        return (
            '[train] updates = %d | train loss = %.2f | exs = %d' %
            (self.model.updates, self.model.train_loss.avg, self.n_examples)
            )








