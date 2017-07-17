import parlai.core.build_data as build_data
import codecs
import os
from .benchmark_reader import Benchmark
import re

def camel_case_split(identifier):
    # https://stackoverflow.com/a/29920015/4507677
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return ' '.join([m.group(0) for m in matches])

def create_fb_format(dpath):
    print('[building fbformat]')
    # TODO we haven't built anything for the test set yet.
    # hopefully it will be in more or less the same format minus the lexics
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
                    # TODO make the removal underscores and camelCase optional?
                    triple.s = triple.s.replace('_',' ')
                    triple.p = camel_case_split(triple.p)
                    triple.o = triple.o.replace('_',' ')
                    triples += '\\n' + triple.s + '\\t' + triple.p +\
                                '\\t' + triple.o
                target = lex.lex
                handle = ftrain
                if dataset == 'dev':
                    handle = fvalid
                handle.write('1 ' + triples + '\t' + target + '\n')
    ftrain.close()
    fvalid.close()

def build(opt):
    dpath = os.path.join(opt['datapath'], 'WebNLG')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
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
        build_data.mark_done(dpath, version_string=version)