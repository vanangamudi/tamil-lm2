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
Sample   =  namedtuple('Sample', ['id', 'orig_sentence', 'sentence', 'pair', 'existence', 'max_token_len'])

def load_data(config,
              filename='../dataset/lm_lengthsorted.txt',
              max_sample_size=None):
    
    samples = []
    skipped = 0

    input_vocab = Counter()
    output_vocab = Counter()
    
    try:
        log.info('processing file: {}'.format(filename))
        text_file = open(filename).readlines()[:config.HPCONFIG.max_samples]
        for i, l in tqdm(enumerate(text_file),
                            desc='processing {}'.format(filename)):

            orig_sentence = l.strip().split()
            if len(orig_sentence) > 20:
                continue

            #print('===========')
            grouped_token_sentence = []
            token = []
            token.append(orig_sentence[0])
            i = 1
            while i < len(orig_sentence):

                if orig_sentence[i]:
                    token.append(orig_sentence[i])
                        
                #print(orig_sentence[i])
                if orig_sentence[i].endswith('@@'):
                    #print('endswith @@')
                    pass

                else:
                    #print('not endswith @@')
                    if token:
                        grouped_token_sentence.append(token)
                    #print(token)
                    #print(grouped_token_sentence)
                    token = []

                i += 1

            if token:
                grouped_token_sentence.append(token)
                #print(grouped_token_sentence)
            
            sentence = grouped_token_sentence
            if len(sentence) < 3:
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
                    
                    samples.append(
                        Sample('{}.{}'.format(i, center_word_pos),
                               orig_sentence,
                               sentence,
                               (center_word, sentence[context_word_pos]),
                               True,
                               max([len(t) for t in  (center_word, sentence[context_word_pos])]) #will be used in batchop for padding
                        )
                    )

                for w in range(0, config.HPCONFIG.window_size - 1):
                    context_word_pos = center_word_pos - w
                    # make soure not jump out sentence
                    if (context_word_pos < 0
                        or context_word_pos >= len(sentence)
                        or center_word_pos == context_word_pos):
                        continue
                    
                    samples.append(
                        Sample('{}.{}'.format(i, center_word_pos),
                               orig_sentence,
                               sentence,
                               (center_word, sentence[context_word_pos]),
                               False,
                               max([len(t) for t in  (center_word, sentence[context_word_pos])])
                        )
                    )
                    
                for w in range(config.HPCONFIG.window_size + 1, len(sentence)):
                    context_word_pos = center_word_pos + w
                    # make soure not jump out sentence
                    if (context_word_pos < 0
                        or context_word_pos >= len(sentence)
                        or center_word_pos == context_word_pos):
                        continue
                    
                    samples.append(
                        Sample('{}.{}'.format(i, center_word_pos),
                               orig_sentence,
                               sentence,
                               (center_word, sentence[context_word_pos]),
                               False,
                               max([len(t) for t in  (center_word, sentence[context_word_pos])])
                        )
                    )
                    
            if  max_sample_size and len(samples) > max_sample_size:
                break

    except:
        skipped += 1
        log.exception('{}'.format(l))

    print('skipped {} samples'.format(skipped))
    
    log.info('building input_vocabulary...')
    for sample in tqdm(samples):
        for tokens in sample.sentence:
            input_vocab.update(tokens)

        output_vocab.update([sample.existence])

    #pivot = int(len(samples) * config.CONFIG.split_ratio)
    #train_samples, test_samples = samples[:pivot], samples[pivot:]
    train_samples, test_samples = samples, []
    input_vocab = Vocab(input_vocab, special_tokens=VOCAB, freq_threshold=50)
    output_vocab = Vocab(output_vocab)
    return Dataset(filename,
                   (train_samples, test_samples),
                   input_vocab = input_vocab,
                   output_vocab = output_vocab)

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
    max_len = max([d.max_token_len for d in datapoints])
    
    word1 = []
    word2 = []
    existence = []
    for d in datapoints:
        w1, w2 = d.pair
        word1.append([VOCAB[i] for i in w1])
        word2.append([VOCAB[i] for i in w2])

        existence.append(d.existence)
        
    word1 = LongVar(config, pad_seq(word1))
    word2 = LongVar(config, pad_seq(word2))
    
    existence = LongVar(config, existence)
    
    batch = indices, (word1, word2), existence
    return batch

