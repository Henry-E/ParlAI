# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Base script for running official ConvAI2 validation eval for f1.
This uses a the version of the dataset which does not contain candidates.
Leaderboard scores will be run in the same form but on a hidden test set.
"""

import os
import subprocess
import random
import sys

from parlai.scripts.eval_model import eval_model, setup_args as base_setup_args
from parlai.core.agents import Agent

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class HenryAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        opennmt_path = os.path.expanduser(opt['opennmt_path'])
        sys.path.insert(0, opennmt_path)
        from onmt.translate.translation_server import TranslationServer
        self.model = TranslationServer()
        self.model.start(opt['model_config_file_name'])
        self.persona = ''
        self.historical_utterances = ''
        self.this_turn_history = ''
        self.next_turn_history = ''
        self.this_thread_id = str(random.randint(100, 100000))


    # def batch_act(self, observations):
    #     import ipdb; ipdb.set_trace()
    #     pass
    #     batch_reply = []
    #     for i, _ in enumerate(observations):
    #         batch_reply.append({'id': self.getID(),
    #                              'text': 'hello'})
    #     return batch_reply

    def act(self):
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
        model_input = [{'id':0, 'src':source_sequence}]
        with HiddenPrints():
            model_output = self.model.run(model_input)
        output_formatted = {'id': self.getID(),
                            'text': model_output[0][0]}
        # call batch_act with batch size one
        # return a dict with {'id': self.getID(), 'text': text_from_onmt}
        return output_formatted


def setup_args(parser=None):
    parser = base_setup_args(parser)
    parser.set_defaults(
        task='convai2:self:no_cands',
        datatype='valid',
        hide_labels=False,
        dict_tokenizer='split',
        metrics='f1',
    )
    return parser


def eval_f1(opt, print_parser):
    report = eval_model(opt, print_parser)
    print('============================')
    print('FINAL F1: ' + str(report['f1']))


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(model='projects.convai2.submission.eval_f1:HenryAgent')
    parser.add_argument('--opennmt_path', \
                        default='~/downloads/OpenNMT-py/',
                        help='path to the main directory of the regular opennmt-py repo')
    parser.add_argument('--model_path', \
                        default='~/projects/convai2/convai2_models/experiments/training_models_11h51m_Thu_30-08-2018_51111/train/hosea_16h01m_Mon_03-09-2018/_step_200.pt',
                        help='path to the model file downloaded from gdrive')
    parser.add_argument('--model_config_file_name', \
                        default='/home/henrye/downloads/Henry_ParlAI/projects/convai2/submission/conf.json')
    # try with --numthreads N to go fast
    opt = parser.parse_args()
    report = eval_f1(opt, parser)
