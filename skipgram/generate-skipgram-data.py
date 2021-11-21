import os
import re
import sys
import glob
import argparse
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from anikattu.debug import memory_consumed

from functools import partial
from collections import namedtuple, defaultdict, Counter

import pickle

from tqdm import tqdm
from anikattu.vocab import Vocab

import multiprocessing as mp
from bloom_filter import BloomFilter

VOCAB =  ['PAD', 'UNK', 'GO', 'EOS']
PAD = VOCAB.index('PAD')
UNK = VOCAB.index('UNK')

PUNCT_SYMBOLS = '/,<>:;\'"[]{}\|!@#$%^&*()_+-=~`'

def remove_punct_symbols(sentence):
    for i in PUNCT_SYMBOLS:
        #print(sentence)
        sentence = sentence.replace(i, ' ')

    return sentence

def count_UNKS(sentence, vocab):
    return sum(
        [1 for i in sentence if vocab[i] == vocab['UNK']]
    )

def vocab_filter(sentence, vocab):
    return [i if vocab[i] != vocab['UNK'] else 'UNK' for i in sentence ]

def build_vocab(filepath):
    input_vocab= Counter()
    text_file = open(filepath)

    log.info('building input_vocabulary...')
    sentences = set()
    for i, l in tqdm(enumerate(text_file),
                        desc='processing {}'.format(filepath)):

        if i % 1000000 == 0:
            print('memory consumed: ', memory_consumed())

        sentence = remove_punct_symbols(l)
        sentence = sentence.strip().split()
        if len(sentence):
            input_vocab.update(sentence)

    return input_vocab

def load_vocab(args, freq_threshold, text_filepath):
    try:
        vocab = {}
        with open(args.output_dir + '/freq_dict.tsv') as vocab_entries:
            for line in tqdm(vocab_entries):
                try:
                    token, count = line.split('\t')
                    count = int(count)
                    if count >  freq_threshold:
                        vocab[token] = count
                except:
                    log.exception('improper vocab entry: {}'.format(line))
                
    except:
        log.exception('filepath')
        vocab = build_vocab(text_filepath)
        
        with open(args.output_dir + '/freq_dict.tsv', 'w') as vocab_path:
            for k, v in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
                vocab_path.write('{}\t{}\n'.format(k, v))
                
         
    return Vocab(vocab,
                 special_tokens = VOCAB,
                 freq_threshold = int(freq_threshold))
            
def print_memory():
    print('{:0.3}GB'.format(memory_consumed() / 1000 / 1000 / 1000 ))

def find_line_seek_positions(filepath):

    line_pos = []
    file_pos = 0
    line_pos.append(file_pos)
    with open(filepath, 'rb') as f:
        for line in tqdm(f):
            file_pos += len(line)
            line_pos.append(file_pos)
    
    return line_pos
    
def process_file_chunk(args, filepath, vocab, dataset_output_file, bpos, epos):
    
    dataset_output_file = open(dataset_output_file, 'w')
    text_file = open(filepath)
    text_file.seek(bpos)
    bloom_filter = BloomFilter(max_elements=10000000, error_rate=0.1)
    
    i = 0
    sentence = text_file.readline()
    pbar = tqdm(total=epos-bpos)
    while sentence:        
        i += 1
        if i % 10000000 == 0:
            print_memory()
            
        if text_file.tell() >= epos:
            break

        log.debug('sentence :'.format(sentence))
        pbar.update(text_file.tell() - bpos)
        sentence = remove_punct_symbols(sentence)
        sentence = sentence.strip().split()

        if len(sentence) < 2:
            log.debug('sentence length < 2, {}'.format(' '.join(sentence)))
            sentence = text_file.readline()
            continue

        unk_ratio = float(count_UNKS(sentence, vocab))/len(sentence)

        log.debug('===')
        log.debug('{} {}'.format(i, pformat(sentence)))

        sentence =  [vocab[i] for i in sentence]
        log.debug('{} {}'.format(i, pformat(sentence)))

        if unk_ratio > 0.7:
            log.debug('unk ratio is heavy: {}'.format(unk_ratio))
            sentence = text_file.readline()
            continue

        for center_word_pos, center_word in enumerate(sentence):
            for w in range(-args.window_size,
                            args.window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if (context_word_pos < 0
                    or context_word_pos >= len(sentence)
                    or center_word_pos == context_word_pos):
                    continue

                pair = (center_word, sentence[context_word_pos])
                if pair[0] != UNK and pair[1] != UNK:
                    if not pair in bloom_filter:
                        log.debug('new pair: {}'.format(pair))
                        dataset_output_file.write(
                            '\t'.join(
                                str(s) for s in  [
                                    #'{}.{}'.format(i, center_word_pos),
                                    center_word,
                                    sentence[context_word_pos],
                                    #vocab[center_word],
                                    #vocab[sentence[context_word_pos]],
                                    #(context_word_pos - center_word_pos),
                                ])

                            + '\n')

                    bloom_filter.add(pair)

        sentence = text_file.readline()
        
def load_data(args):
    
    samples = []
    skipped = 0
    
    input_vocab = Counter()

    
    try:
        log.info('processing file: {}'.format(args.input_))
        
        vocab = load_vocab(args, 100, args.input_)
        text_file = open(args.input_)
        UNK = vocab['UNK']

        line_pos_file = args.output_dir + '/line_pos.pkl'
        if os.path.exists(line_pos_file):
            line_pos = pickle.load(open(line_pos_file, 'rb'))
        else:
            line_pos = find_line_seek_positions(args.input_)
            pickle.dump(line_pos, open(line_pos_file, 'wb'))
                
        log.info('number of lines in input: {}'.format(len(line_pos)))
        log.info('number of lines in input: {}'.format(line_pos[:100]))

        """
        process_file_chunk(args,
                           args.input_,
                           vocab,
                           args.output_pattern.format(0),
                           0,
                           100000,)
        """
        
        results = []
        pool = mp.Pool(processes=2)
        for i, bpos in enumerate(range(0, len(line_pos), args.chunk_size)):


            bpos = line_pos[bpos]
            if bpos + args.chunk_size > len(line_pos):
                epos = line_pos[-1]
            else:
                epos = line_pos[bpos + args.chunk_size]
                
            log.info('processing from {} to {}'.format(bpos, epos))
            
            results.append(pool.apply_async(
                process_file_chunk,
                args=(args,
                      args.input_,
                      vocab,
                      args.output_pattern.format(i),
                      bpos,
                      epos,)
            ))

        output = [p.get() for p in results]
        print(output)

    except:
        log.exception('===')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SkipGram model for training tamil language model')
    parser.add_argument('-i','--input',
                        help='path to the text corpus file',
                        dest='input_')

    parser.add_argument('-o','--output-pattern',
                        help='path pattern to the output files',
                        dest='output_pattern')
    
    parser.add_argument('-d','--output-dir',
                        help='path pattern to the output files',
                        dest='output_dir')

    parser.add_argument('-w','--window-size',
                        type=int,
                        help='path pattern to the output files',
                        dest='window_size')

    parser.add_argument('-c','--chunk_size',
                        type=int,
                        help='path pattern to the output files',
                        dest='chunk_size')
    args = parser.parse_args()
    print(args)

    os.makedirs(os.path.dirname(args.output_pattern), exist_ok=True)
    load_data(args)
