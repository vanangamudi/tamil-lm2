
import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from tqdm import tqdm

class Vocab:
    def __init__(self, vocab, special_tokens=[],
                 max_size=None, sort=False, sort_key=None,
                 freq_threshold=1, tokens=None):

        log.info('Constructiong vocabuluary object...')
        self.freq_dict = vocab
        self.special_tokens = special_tokens
        if isinstance(freq_threshold, int):
            vocab = {w:c for w, c in tqdm(vocab.items(), 'Vocab:thresholding...')
                     if c >= freq_threshold}
        else:
            l, h = freq_threshold
            vocab = {w:c for w, c in tqdm(vocab.items(), 'Vocab:thresholding...')
                     if c <= l or c >= h}

        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

        if max_size: vocab = vocab[:max_size]
        
        vocab = [ w for w,c in vocab]
        index2word = vocab
        if tokens  :  index2word = tokens
        if sort:
            index2word = sorted(index2word, key=sort_key)

        self.index2word = special_tokens + index2word
        self.word2index = { w:i for i, w in enumerate(self.index2word) }

        log.info('number of word in index2word and word2index: {} and {}'
                 .format(len(self.index2word), len(self.word2index)))

        
    def __getitem__(self, key):
        if type(key) == str:
            try:
                return self.word2index[key]
            except:
                #log.exception('==========')
                return self.word2index['UNK']
        else: #type(key) == int:
            return self.index2word[key]
        

    def __len__(self):
        return len(self.index2word)

    def extend(self, words):        
        self.word2index.update(
            {
                w  :   i + len(self.index2word)
                for i, w in enumerate(words)
            })
        
        self.index2word += words

    def write_to_file(self, filepath, delim='|'):
        lines = ['{}{}{}'.format(k, delim, v) for k, v in self.freq_dict.items()]
        with open(filepath + '.freq_dict.csv', 'w') as f:
            f.write('\n'.join(lines))

        with open(filepath + '.index2word.csv', 'w') as f:
            f.write('\n'.join(self.index2word))

    def load_from_file(self, filepath, delim='|'):
        self.freq_dict = {}
        with open(filepath + '.freq_dict.csv') as f:
            for line in f.readlines():
                word, freq = line.strip().split(delim)
                self.freq_dict[word] = int(freq)

        with open(filepath + '.index2word.csv') as f:
            self.index2word = f.read().splitlines()

        
