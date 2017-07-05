import os
import sys
import logging

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def add_cmdline_args(parser):
    # Runtime environment
    agent = parser.add_argument_group('WebNLG Arguments')
    agent.add_argument('--no_cuda', type='bool', default=False)
    agent.add_argument('--gpu', type=int, default=-1)
    agent.add_argument('--random_seed', type=int, default=1013)

    # Basics
    agent.add_argument('--embedding_file', type=str, default=None,
                        help='File of space separated embeddings: w e1 ... ed')
    agent.add_argument('--pretrained_model', type=str, default=None,
                        help='Load dict/features/weights/opts from this file')
    agent.add_argument('--log_file', type=str, default=None)

    # Model details
    agent.add_argument('--fix_embeddings', type='bool', default=True)
    agent.add_argument('--tune_partial', type=int, default=0,
                        help='Train the K most frequent word embeddings')
    agent.add_argument('--embedding_dim', type=int, default=300,
                        help=('Default embedding size if '
                              'embedding_file is not given'))
    agent.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size of RNN units')
    agent.add_argument('--rnn_type', type=str, default='gru',
                        help='RNN type: gru (default), lstm, or rnn')

    # Optimization details
    agent.add_argument('--valid_metric', type=str,
                        choices=['accuracy', 'f1'], default='f1',
                        help='Metric for choosing best valid model')
    agent.add_argument('--max_len', type=int, default=15,
                        help='The max span allowed during decoding')
    agent.add_argument('--rnn_padding', type='bool', default=False)
    agent.add_argument('--display_iter', type=int, default=10,
                        help='Print train error after every \
                              <display_iter> epoches (default 10)')
    agent.add_argument('--dropout_emb', type=float, default=0.4,
                        help='Dropout rate for word embeddings')
    agent.add_argument('--dropout_rnn', type=float, default=0.4,
                        help='Dropout rate for RNN states')
    agent.add_argument('--dropout_rnn_output', type='bool', default=True,
                        help='Whether to dropout the RNN output')
    agent.add_argument('--optimizer', type=str, default='adamax',
                        help='Optimizer: sgd or adamax (default)')
    agent.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                        help='Learning rate for SGD (default 0.1)')
    agent.add_argument('--grad_clipping', type=float, default=10,
                        help='Gradient clipping (default 10.0)')
    agent.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (default 0)')
    agent.add_argument('--momentum', type=float, default=0,
                        help='Momentum (default 0)')

def set_defaults(opt):
    # Embeddings options
    if opt.get('embedding_file'):
        if not os.path.isfile(opt['embedding_file']):
            raise IOError('No such file: %s' % args.embedding_file)
        with open(opt['embedding_file']) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        opt['embedding_dim'] = dim
    elif not opt.get('embedding_dim'):
        raise RuntimeError(('Either embedding_file or embedding_dim '
                            'needs to be specified.'))