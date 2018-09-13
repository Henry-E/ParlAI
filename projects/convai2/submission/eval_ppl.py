# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Base script for running official ConvAI2 validation eval for perplexity.
This uses a the version of the dataset which does not contain candidates.
Leaderboard scores will be run in the same form but on a hidden test set.

The official vocabulary for the competition is based on using the
"split_tokenize" method on in the ParlAI core dictionary (parlai/core/dict.py)
and is built on the training and validation sets of the "convai2" task.
This dictionary contains a total of 19304 tokens. The test set contains some
tokens which are not in this dictionary--this tokens will not be provided, but
we will also *SKIP* calculating perplexity on these tokens. The model should
still produce a good guess for the remaining tokens in the sentence, so
handling unknown words or expanding the vocabulary with pre-trained or
multitasked embeddings are legitimate strategies that may or may not impact the
score of the models.

Note that this tokenizer will also be used during the perplexity evaluation:
the model will be asked to predict one word at a time according to this
tokenizer's parsing of the text.

This requires agents to implement the following function:

def next_word_probability(self, partial_out):
    Return probability distribution over next words given a partial true output.
    This is used to calculate the per-word perplexity.

    Arguments:
    partial_out -- list of previous "true" words

    Returns a dict, where each key is a word and each value is a probability
    score for that word. Unset keys assume a probability of zero.

    e.g.
    {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
"""
import os
import json
import random
import subprocess

from parlai.core.agents import Agent

from parlai.scripts.eval_ppl import eval_ppl as run_eval_ppl, setup_args as setup_ppl_args
from projects.convai2.build_dict import build_dict


def setup_args(parser=None):
    parser = setup_ppl_args(parser)
    parser.set_defaults(
        task='convai2:self:no_cands',
        datatype='valid',
        dict_tokenizer='split',
    )
    return parser


class HenryEntry(Agent):
    """This is an example entry which tries to use the RepeatLabelAgent.
    Since no labels are given to the model, it will guess something useless.

    It builds the official dictionary first, so that it can provide a minimum
    probablity for each word as well as use the official tokenizer.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            # build official eval dictionary
            self.dict = build_dict()
        else:
            # only build dict once
            self.dict = shared['dict']
        # import ipdb;ipdb.set_trace()
        max_freq = self.dict.max_freq()
        # set probability of each word, skipping the invalid words like __NULL__
        # (which have frequency more than max_freq)
        self.freqs = {k: f for k, f in self.dict.freqs().items() if f <= max_freq}
        self.persona = ''
        self.historical_utterances = ''
        self.this_turn_history = ''
        self.next_turn_history = ''
        self.this_thread_id = str(random.randint(100, 100000))


    def share(self):
        shared = super().share()
        # share dict with other threads instead of rebuilding in each
        shared['dict'] = self.dict
        return shared

    def next_word_probability(self, partial_out):
        """Example implementation of next word probability."""
        # import ipdb; ipdb.set_trace()
        obs = self.observation
        chat = obs['text'].split('\n')
        persona = [line.strip().split('your persona: ')[1][:-1] for line in
                   chat if 'your persona: ' in line]
        # add the persona delimiter before and after each persona
        personas_combined = '{1}{0}{1}'.format(' _ps_ '.join(persona),
                                               ' _ps_ ')
        if persona:
            self.persona = personas_combined.strip()
        # shus = start historical utterance source
        self.this_turn_history = ' _shus_ ' + chat[-1] + ' _ehus_' + \
                                ' _shut_ ' + obs['eval_labels'][0] + ' _ehut_'
        # kind of a random work around because history + personas aren't
        # available in the observation, only from previous observations
        if 'your persona:' in obs['text']:
            self.historical_utterances = ''
        if 'your persona: ' not in obs['text'] and \
                self.this_turn_history != self.next_turn_history:
            # import ipdb; ipdb.set_trace()
            self.historical_utterances += self.next_turn_history
        self.next_turn_history = self.this_turn_history
        source_utterance = chat[-1]
        source_sequence = self.historical_utterances + ' ' + \
            self.persona + ' _su_ ' + source_utterance
        source_sequence = source_sequence.strip()
        # import ipdb; ipdb.set_trace()

        this_dir = os.path.dirname(__file__)
        temp_eval_dir = os.path.join(this_dir, 'temp_eval_files')
        source_file_name = os.path.join(temp_eval_dir,
                                        'source_utterance_' +
                                        self.this_thread_id +
                                        '.txt')
        # write a single utterance, floating in perfume, served in a man's hat
        with open(source_file_name, 'w') as out_file:
            out_file.write(source_sequence)
        target_file_name = os.path.join(temp_eval_dir,
                                        'target_utterance_' +
                                        self.this_thread_id +
                                        '.txt')
        if not partial_out:
            # opennmt can't handle an empty target file
            partial_out = ['i', '_this_', 'i', '_is_', 'i',  '_blank_', 'i', '_surely_']
        with open(target_file_name, 'w') as out_file:
            out_file.write(' '.join(partial_out))
        probabilities_file_name = os.path.join(temp_eval_dir, 'probabilities_' +
                                               self.this_thread_id + '.txt')
        translate_module_file_name = \
            os.path.join(os.path.expanduser(self.opt['opennmt_path']), 'translate.py')
        model_file_name = os.path.expanduser(self.opt['model_path'])
        args = ['python', translate_module_file_name, '-model',
                model_file_name, '-src', source_file_name, '-tgt',
                target_file_name, '-output', probabilities_file_name,
                '-beam_size', '1', '-batch_size', '1', '-replace_unk',
                '-gpu', '0']
        # print(' '.join(args))

        # import ipdb; ipdb.set_trace()
        _ = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with open(probabilities_file_name) as in_file:
            probs = json.load(in_file)
        probs['__end__'] = probs.pop('</s>')
        probs_out = {token: probs[token] if token in probs else 1e-7 for token
                     in list(self.dict.keys())}

        # import ipdb; ipdb.set_trace()

        # # initialize probabilities with inverse word frequency
        # freqs = self.freqs.copy()

        # # increase likelihood of predicting input words
        # tokens = self.dict.tokenize(obs.get('text', ''))
        # for t in tokens:
        #     if t in freqs:
        #         freqs[t] += 10000
        return probs_out


def eval_ppl(opt):
    return run_eval_ppl(opt, build_dict)


if __name__ == '__main__':
    parser = setup_args()
    # example model just uses word frequencies
    parser.set_defaults(model='projects.convai2.submission.eval_ppl:HenryEntry')
    parser.add_argument('--opennmt_path', \
                        default='~/downloads/Henry_OpenNMT-py/',
                        help='path to the main directory of the cloned custom opennmt-py repo')
    parser.add_argument('--model_path', \
                        default='~/projects/convai2/convai2_models/experiments/training_models_11h51m_Thu_30-08-2018_51111/train/hosea_16h01m_Mon_03-09-2018/_step_200.pt',
                        help='path to the model file downloaded from gdrive')
    # try with --numthreads N to go fast
    opt = parser.parse_args()
    eval_ppl(opt)
    if opt['model'] == 'projects.convai2.eval_ppl:WordFrequencyEntry':
        print('This run just used the example filler model. To get better '
              'results, try implementing your own!')
