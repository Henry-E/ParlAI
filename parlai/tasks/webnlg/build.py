import parlai.core.build_data as build_data
import codecs
import os
from .benchmark_reader import Benchmark
import ipdb

def create_fb_format(dpath):
    print('[building fbformat]')
    # TODO there's a lot more information that could accompany
    # this data. There's also some data cleaning that can only
    # take place at this stage. We're waiting to see what is 
    # necessary before going any further
    ftrain = open(os.path.join(dpath, 'train.txt'), 'w')
    fvalid = open(os.path.join(dpath, 'valid.txt'), 'w')
    for dataset in {'train', 'dev'}:
        data_path = os.path.join(dpath, dataset)
        files = []
        for (dirpath, dirnames, filenames) in os.walk(data_path):
            if filenames:
                for filename in filenames:
                    files.append(os.path.join(dirpath, filename))
        parsed_xml = Benchmark()
        parsed_xml.fill_benchmark(files)
        for entry in parsed_xml.entries:
            tripleset = entry.modifiedtripleset
            lexics = entry.lexs
            category = entry.category
            for lex in lexics:
                triples = ''
                for triple in tripleset.triples:
                    triples += triple.s + ' __s__ ' + triple.p +\
                    ' __p__ ' + triple.o + ' __o__ '
                target = lex.lex
                handle = ftrain
                if dataset == 'dev':
                    handle = fvalid
                handle.write('1 ' + triples + '\t' + target + '\n')
    ftrain.close()
    fvalid.close()

def build(opt):
    dpath = os.path.join(opt['datapath'], 'WebNLG')

    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname = 'challenge_data_train_dev.zip'
        url = 'http://talc1.loria.fr/webnlg/stories/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)
        # ipdb.set_trace()
        # # delexicalise it and other stuff
        # dpext = os.path.join(dpath, 'delexicalised')
        # # TODO fix all the issues with linking 
        # dpath = dpath + '/'
        # webnlg_baseline_input.main(dpath, dpext)

        # file_ext = os.path.join(dpext, '{}-webnlg-all-delex.{}')
        create_fb_format(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath)