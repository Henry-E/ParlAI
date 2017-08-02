import parlai.core.build_data as build_data
import codecs
import os
from .benchmark_reader import Benchmark
import re
from nltk.tokenize.moses import MosesTokenizer
from parlai.agents.webnlg.apply_bpe import BPE


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
                triples = []
                for triple in tripleset.triples:
                    # TODO make the removal underscores and camelCase optional?
                    triple.s = triple.s.replace('_',' ')
                    triple.p = camel_case_split(triple.p).lower()
                    triple.o = triple.o.replace('_',' ')
                    triples += [triple.s + '\\t' + triple.p +\
                                '\\t' + triple.o]
                target = lex.lex
                handle = ftrain
                triples = '\\n'.join(triples)
                if dataset == 'dev':
                    handle = fvalid
                handle.write('1 ' + triples + '\t' + target + '\n')
    ftrain.close()
    fvalid.close()

def create_target_source_files(input_file):
    bpe_codes = open('/home/henrye/data/word_vectors/subWordFastText/full_wikipedia.en.20000.bpe.codes')
    encoder = BPE(bpe_codes)
    word_tok = MosesTokenizer(no_escape=True)
    def tokenize(text):
        """Uses nltk Treebank Word Tokenizer for tokenizing words within
        sentences.
        """
        word_tokens = word_tok.tokenize(text)
        sub_word_tokens = encoder.segment(' '.join(word_tokens))
        return sub_word_tokens
    outpath = os.path.dirname(input_file)
    file_name = os.path.basename(os.path.splitext(input_file)[0])
    target_file = open(os.path.join(outpath, file_name + "-target.txt"), 'w')
    source_file = open(os.path.join(outpath, file_name + "-source.txt"), 'w')
    with codecs.open(input_file, 'r') as f:
        for line in f:
            l = line.split('\t')
            triples_raw = l[0].strip('1 ')
            targets_raw = l[1].strip('\n')
            triples_split = [triple.split('\\t') for triple in triples_raw.split('\\n')]
            triples = ''
            for i, triple in enumerate(triples_split):
                for j, sub_pred_obj in enumerate(triple):
                    triples += tokenize(sub_pred_obj)
                    if len(triple) == 3 and j < 2:
                        triples += ' __PREDICATE__ '
                triples += ' __TRIPLE__ '
            triples = triples.strip()
            targets = tokenize(targets_raw).strip()
            source_file.write(triples + '\n')
            target_file.write(targets + '\n')
    target_file.close()
    source_file.close()

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
        create_fb_format(dpath)
        print('[building source and target files for OpenNMT]')
        create_target_source_files(os.path.join(dpath, "train.txt"))
        create_target_source_files(os.path.join(dpath, "valid.txt"))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)