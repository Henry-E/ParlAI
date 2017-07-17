from parlai.core.fbdialog_teacher import FbDialogTeacher
from .build import build

import copy
import os

def unescape(s):
    """Revert escaped characters back to their special version.
    For example, \\n => newline and \\t => tab
    """
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

def _path(opt, filtered):
    # Build the data if it doesn't exist.
    build(opt)
    # TODO find out why there would ever be a colon in the datatype argument
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'WebNLG',
                        dt + filtered + '.txt')


class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        opt['cands_datafile'] = None
        super().__init__(opt, shared)

    def setup_data(self, path):
        """Reads data in the fbdialog format.

        Returns ``((x,y,r,c), new_episode?)`` tuples.

        ``x`` represents a query, ``y`` represents the labels, ``r`` represents any reward,
        and ``c`` represents any label_candidates.

        The example above will be translated into the following tuples:

        ::

            x: 'Sam went to the kitchen\\nPat gave Sam the milk\\nWhere is the milk?'
            y: ['kitchen']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = True (this is the first example in the episode)


        ::

            x: 'Sam went to the hallway\\nPat went to the bathroom\\nWhere is the
                milk?'
            y: ['hallway']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = False (this is the second example in the episode)
        """
        print("[loading fbdialog data:" + path + "]")
        with open(path) as read:
            start = True
            x = ''
            reward = None
            dialog_index = 0
            for line in read:
                line = line.strip()
                if len(line) == 0:
                    continue

                # first, get conversation index -- '1' means start of episode
                space_idx = line.find(' ')
                conv_id = line[:space_idx]

                # split line into constituent parts, if available:
                # x<tab>y<tab>reward<tab>label_candidates
                # where y, reward, and label_candidates are optional
                split = line[space_idx + 1:].split('\t')

                # remove empty items and strip each one
                for i in range(len(split)):
                    word = split[i].strip()
                    if len(word) == 0:
                        split[i] = ''
                    else:
                        # this line represents the only difference between 
                        # this and fbdialog
                        split[i] = unescape(word)
                # Empty reward string same as None
                if len(split) > 2 and split[2] == '':
                    split[2] = None

                # now check if we're at a new episode
                if conv_id == '1':
                    dialog_index += 1
                    x = x.strip()
                    if x:
                        yield [x, None, reward], start
                    start = True
                    # start a new episode
                    if self.cloze:
                        x = 'Fill in the blank in the last sentence.\n{x}'.format(
                            x=split[0]
                        )
                    else:
                        x = split[0]
                else:
                    if x:
                        # otherwise add current x to what we have so far
                        x = '{x}\n{next_x}'.format(x=x, next_x=split[0])
                    else:
                        if len(split) > 2:
                            reward = split[2]
                        x = split[0]

                if len(split) > 1 and split[1]:
                    # only generate an example if we have a y
                    split[0] = x
                    # split labels
                    split[1] = split[1].split('|')
                    if len(split) > 3:
                        # split label_candidates
                        split[3] = split[3].split('|')
                    if start:
                        yield split, True
                        start = False
                    else:
                        yield split, False
                    # reset x in case there is unlabeled data still left
                    x = ''
                    reward = None
