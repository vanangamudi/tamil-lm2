import os
import re
import sys
import glob
import pickle
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

from functools import partial
from collections import namedtuple, defaultdict, Counter


from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.datafeed import DataFeed
from anikattu.dataset import Dataset
from anikattu.utilz import tqdm, ListTable
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq
from anikattu.debug import memory_consumed
from nltk.tokenize import WordPunctTokenizer

from bloom_filter import BloomFilter

word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize


VOCAB =  ['PAD', 'UNK', 'GO', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""
Sample   =  namedtuple('Sample',
                       ['id',
                        #'sentence',
                        'word',
                        'context'])

PUNCT_SYMBOLS = '/,<>:;\'"[]{}\|!@#$%^&*()_+-=~`'

def remove_punct_symbols(sentence):
    for i in PUNCT_SYMBOLS:
        sentence = sentence.replace(i, ' ')

    return sentence

def count_UNKS(sentence, vocab):
    return sum(
        [1 for i in sentence if vocab[i] == vocab['UNK']]
    )

def vocab_filter(sentence, vocab):
    return [i if vocab[i] != vocab['UNK'] else 'UNK' for i in sentence ]

def build_vocab(config, filename):
    input_vocab= Counter()
    text_file = open(filename)

    log.info('building input_vocabulary...')
    sentences = set()
    for i, l in tqdm(enumerate(text_file),
                        desc='processing {}'.format(filename)):

        if i % 1000000 == 0:
            print('memory consumed: ', memory_consumed())

        sentence = remove_punct_symbols(l)
        sentence = sentence.strip().split()
        if len(sentence):
            input_vocab.update(sentence)

        if i > config.HPCONFIG.max_samples:
            break

        if config.CONFIG.DEBUG and i > 100000:
            break

    return input_vocab

def load_vocab(config):
    vocab = {}
    with open(config.ROOT_DIR + '/freq_dict.tsv') as vocab_entries:
        for line in tqdm(vocab_entries, desc='loading vocab...'):
            try:
                token, count = line.split('\t')
                vocab[token] = int(count)
            except:
                log.exception('improper vocab entry: {}'.format(line))
    
    return Vocab(vocab,
                 special_tokens = VOCAB,
                 freq_threshold = config.HPCONFIG.freq_threshold)
            
def print_memory():
    print('{:0.3}GB'.format(memory_consumed() / 1000 / 1000 / 1000 ))
          
def load_data(config,
              filepath,
              delim):
    samples = []
    skipped = 0
    
    vocab = load_vocab(config)
    
    log.info('processing file: {}'.format(filepath))
    dataset_size = 0
    dataset_size_path = config.ROOT_DIR + '/dataset_size.pkl'
    if os.path.exists(dataset_size_path):
        dataset_size = pickle.load(open(dataset_size_path, 'rb'))
    else:
        with open(filepath) as f:
            for line in tqdm(f, desc='counting lines'):
                dataset_size += 1
        pickle.dump(dataset_size, open(dataset_size_path, 'wb'))
                
            
    return Dataset(filepath, filepath, dataset_size,  delim, vocab)

# ## Loss and accuracy function
def loss(output, targets, loss_function, *args, **kwargs):
    return loss_function(output, targets)

def accuracy(output, batch, *args, **kwargs):
    indices, sequence, label = batch
    return (output.max(dim=1)[1] == label).sum().float()/float(label.size(0))

def batchop(datapoints, config, *args, **kwargs):
    word, context = datapoints
    word    = LongVar(config, word)
    context = LongVar(config, context)
    
    batch = word, context
    return batch

