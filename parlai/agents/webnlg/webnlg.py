# a first attempt at implementing a custom agent class 

import torch
import os
import numpy as np
import logging
import copy
import unicodedata
from nltk.tokenize.moses import MosesTokenizer

from torch.autograd import Variable
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from . import config
from .model import Triples2TextModel
from .apply_bpe import BPE
from .utils import normalize_text

# ----------------------------------------------------------------------------
# Dictionary.
# ----------------------------------------------------------------------------

class SimpleDictionaryAgent(DictionaryAgent):
    # custom dictionary agent for words segmented using Byte Pair Encoding

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--pretrained_words', type='bool', default=True,
            help='Use only words found in provided embedding_file'
        )
        # TODO turn this into the BPE file?
        group.add_argument(
            '--bpe_codes_file', type=str, default=None,
            help='File of byte pair encoded word segments'
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO possibly create command line arguments for these? but probably
        # unnecessary. We might also experiment with different divider tokens
        self.triple_token = '__TRIPLE__'
        self.predicate_token = '__PREDICATE__'
        
        # TODO when are we ever going to care about whether the dictionary
        # shared or not?
        # if not shared: - from how the init construction is done in dict.py

        if self.opt.get('dict_file') and os.path.isfile(self.opt['dict_file']) \
            or self.opt.get('dict_initpath'):
            pass
        else:
            # set special triple divider word token
            index = len(self.tok2ind)
            self.tok2ind[self.triple_token] = index
            self.ind2tok[index] = self.triple_token
            # fix count for triple token to one billion and four
            self.freq[self.triple_token] = 1000000004

            # set special subject, predicate, object divider word token
            index = len(self.tok2ind)
            self.tok2ind[self.predicate_token] = index
            self.ind2tok[index] = self.predicate_token
            # fix count for divider token to one billion and five
            self.freq[self.predicate_token] = 1000000005
        
        if self.opt.get('bpe_codes_file'):
            bpe_codes = open(self.opt['bpe_codes_file'])
            self.encoder = BPE(bpe_codes)
            # TODO dict instance will break later on if there's no encoder
        
        self.word_tok = MosesTokenizer(no_escape=True)
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

    def tokenize(self, text, building=False):
        """ We decided to remove sentence tokenizing because it was messing 
        up """
        # TODO replace the tabs and new lines with appropriate delimiter
        # tokens, whenever we decide what form they should take
        return (token for token in self._word_tokenize(text, building).split(' '))

    def _word_tokenize(self, text, building=False):
        """Uses nltk Treebank Word Tokenizer for tokenizing words within
        sentences.
        """
        # TODO moses is converting special characters to html or something
        # get it to stop doing that
        word_tokens = self.word_tok.tokenize(text)
        sub_word_tokens = self.encoder.segment(' '.join(word_tokens))
        return sub_word_tokens

class WebnlgAgent(Agent):
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

        # Use an internal flag to switch between handling datasets differently
        self.dataset = 'train'
        self.validation_sentence = ''

    def _init_from_scratch(self):
        self.opt['vocab_size'] = len(self.word_dict)

        print('[ Initializing model from scratch ]')
        self.model = Triples2TextModel(self.opt, self.word_dict)
        self.model.set_embeddings()

    def train(self):
        self.dataset = 'train'

    def evaluate(self):
        self.dataset = 'evaluate'
        self.model.valid_loss.reset()

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
        batch = self._batchify([example])

        # Either train and/or generate
        if self.dataset is 'train':
            self.n_examples += 1 
            self.model.update(batch)
        else:
            self.model.evaluate(batch)
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

        # Towards the end of the validation and test sets the batch can
        # be incomplete
        examples = [self._build_example(obs) for obs in observations if 'text' in obs]

        batch = self._batchify(examples)

        # Either train and/or generate
        if self.dataset is 'train':
            self.n_examples += len(examples)
            self.model.update(batch)
        else:
            # sentence_idxs = self.model.generate(batch)
            # self.validation_sentence = self.word_dict.vec2txt(sentence_idxs)
            self.validation_sentence = ''
            self.model.evaluate(batch)
            # TODO generate

        return batch_reply




    # ------------------------------------------------------------------------
    # Helper functions.
    # ------------------------------------------------------------------------

    def _build_example(self, example, end_idx=1, unk_idx=2, start_idx=3):
        # Add words which appear in input but not in dictionary to dictionary
        # to be used with pointer network later on
        # TODO for efficiency make this only happen during the first epoch
        # TODO possibly make this optional? It doesn't really affect anything
        # by having it as default though
        self.word_dict.extend_dict(example['text'])


        # the delimiter has to be added after the text processing otherwise
        # it gets broken apart as if it were a subword
        triples_split = [triple.split('\t') for triple in example['text'].split('\n')]
        triples = []
        for i, triple in enumerate(triples_split):
            for j, sub_pred_obj in enumerate(triple):
                triples += self.word_dict.txt2vec(sub_pred_obj)
                if len(triple) == 3 and j < 2:
                    triples += [self.word_dict['__PREDICATE__']]
            triples += [self.word_dict['__TRIPLE__']]
        triples = torch.LongTensor(triples)

        if 'labels' in example:
            # watch out, labels is a tuple not a single string
            targets = self.word_dict.txt2vec(example['labels'][0])
            targets = [start_idx] + targets + [end_idx]
            # Replace any OOV indices which appear in targets but not in triples.
            # These are OOV words that were in earlier triples but not this one
            for idx, target in enumerate(targets):
                if target > self.opt['vocab_size'] - 1 and target not in triples:
                    targets[idx] = unk_idx
            targets = torch.LongTensor(targets)
            return triples, targets
        else: 
            print('returning only triples because there\'s no target text availble')
            return triples

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
        if self.dataset is 'train':
            return (
                '[train] updates = %d | train loss = %.4f | exs = %d' %
                (self.model.updates, self.model.train_loss.avg, self.n_examples)
                )
        else:
            return (
                'valid loss = %.4f | exs = %d | sample sentence = %s' %
                (self.model.valid_loss.avg, self.model.valid_loss.count,
                    self.validation_sentence)
                )








