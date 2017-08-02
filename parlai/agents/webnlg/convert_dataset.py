from nltk.tokenize.moses import MosesTokenizer
import codecs
import os
from apply_bpe import BPE

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


create_target_source_files("/home/henrye/projects/ParlAI/data/WebNLG/valid.txt")
create_target_source_files("/home/henrye/projects/ParlAI/data/WebNLG/train.txt")
