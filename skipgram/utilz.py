import os
import re
import sys
import glob
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
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq

from nltk.tokenize import WordPunctTokenizer
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

def load_data(config,
              filename='../dataset/lm_lengthsorted.txt',
              max_sample_size=None):
    
    samples = []
    skipped = 0

    input_vocab = Counter()
    output_vocab = Counter()
    bloom_filter = Counter()
    try:
        log.info('processing file: {}'.format(filename))
        text_file = open(filename).readlines()
        
        log.info('building input_vocabulary...')
        sentences = set()
        for i, l in tqdm(enumerate(text_file[:config.HPCONFIG.max_samples]),
                            desc='processing {}'.format(filename)):

            sentence = remove_punct_symbols(l)
            sentence = sentence.strip().split()
            if len(sentence):
                input_vocab.update(sentence)
                sentences.add(tuple(sentence))

                
        freq_threshold = (config.HPCONFIG.freq_threshold * (float(config.HPCONFIG.max_samples)
                                                            /len(text_file)))
        log.info('freq_threhold: {}'.format(freq_threshold))
        vocab = Vocab(input_vocab,
                      special_tokens = VOCAB,
                      freq_threshold = int(freq_threshold))

        if config.CONFIG.write_vocab_to_file:
            vocab.write_vocab_to_file(config.ROOT_DIR + '/vocab.csv')
        
        for i, sentence in tqdm(enumerate(sentences),
                         desc='processing sentences'):

            if len(sentence) < 2:
                continue
            
            unk_ratio = float(count_UNKS(sentence, vocab))/len(sentence)

            log.debug('===')
            log.debug(pformat(sentence))
            
            sentence =  [i if vocab[i] != vocab['UNK'] else 'UNK' for i in sentence ]
            log.debug(pformat(sentence))

            if unk_ratio > 0.7:
                log.debug('unk ratio is heavy: {}'.format(unk_ratio))
                continue
                
            for center_word_pos, center_word in enumerate(sentence):
                for w in range(-config.HPCONFIG.window_size,
                                config.HPCONFIG.window_size + 1):
                    context_word_pos = center_word_pos + w
                    # make soure not jump out sentence
                    if (context_word_pos < 0
                        or context_word_pos >= len(sentence)
                        or center_word_pos == context_word_pos):
                        continue

                    pair = (center_word, sentence[context_word_pos])
                    if pair[0] != 'UNK' and pair[1] != 'UNK':
                        if not pair in bloom_filter:
                            pass
                            samples.append(
                                Sample('{}.{}'.format(i, center_word_pos),
                                       #sentence,
                                       center_word,
                                       sentence[context_word_pos]
                                )
                            )
                        bloom_filter.update([pair])
                        
            if  max_sample_size and len(samples) > max_sample_size:
                break

    except:
        skipped += 1
        log.exception('{}'.format(l))

    print('skipped {} samples'.format(skipped))

    if config.CONFIG.dump_bloom_filter:
        with open('word_pair.csv', 'w') as F:
            for k,v in bloom_filter.items():
                F.write('|'.join(list(k) + [str(v)]) + '\n')
                    
    #pivot = int(len(samples) * config.CONFIG.split_ratio)
    #train_samples, test_samples = samples[:pivot], samples[pivot:]
    train_samples, test_samples = samples, []

    return Dataset(filename,
                   (train_samples, test_samples),
                   input_vocab = vocab,
                   output_vocab = vocab)

# ## Loss and accuracy function
def loss(output, targets, loss_function, *args, **kwargs):
    return loss_function(output, targets)

def accuracy(output, batch, *args, **kwargs):
    indices, sequence, label = batch
    return (output.max(dim=1)[1] == label).sum().float()/float(label.size(0))


def waccuracy(output, batch, config, *args, **kwargs):
    indices, sequence, label = batch

    index = label
    src = Var(config, torch.ones(label.size()))
    
    acc_nomin = Var(config, torch.zeros(output.size(1)))
    acc_denom = Var(config, torch.ones(output.size(1)))

    acc_denom.scatter_add_(0, index, (label == label).float() )
    acc_nomin.scatter_add_(0, index, (label == output.max(1)[1]).float())

    accuracy = acc_nomin / acc_denom

    #pdb.set_trace()
    return accuracy.mean()

def repr_function(output, batch, VOCAB, LABELS, dataset):
    indices, sequence, label = batch
    results = []
    output = output.max(1)[1]
    output = output.cpu().numpy()
    for idx, c, a, o in zip(indices, sequence, label, output):

        c = ' '.join([VOCAB[i] for i in c]).replace('\n', ' ')
        a = ' '.join([LABELS[a]])
        o = ' '.join([LABELS[o]])
        
        results.append([str(idx), c, a, o, str(a == o) ])
        
    return results

def batchop(datapoints, VOCAB, config, *args, **kwargs):
    indices = [d.id for d in datapoints]
    word = []
    context = []
    
    for d in datapoints:
        word.append   (VOCAB[   d.word])
        context.append(VOCAB[d.context])

    word    = LongVar(config, word)
    context = LongVar(config, context)
    
    batch = indices, word, context
    return batch

