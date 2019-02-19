import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')
import config
from anikattu.logger import CMDFilter
import logging
logging.basicConfig(format=config.FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
import sys


from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.utilz import initialize_task

from model.skipgram_conv2d_embedding import Model as Model
from utilz import load_data, batchop, loss, accuracy, waccuracy

from functools import partial


from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable, dump_vocab_tsv, dump_cosine_similarity_tsv
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq
from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize


import importlib


SELF_NAME = os.path.basename(__file__).replace('.py', '')

import sys
import pickle
import argparse
from matplotlib import pyplot as plt
plt.style.use('ggplot')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SkipGram model for training tamil language model')
    parser.add_argument('-p','--hpconfig',
                        help='path to the hyperparameters config file',
                        default='hpconfig.py', dest='hpconfig')
    parser.add_argument('--log-filters',
                        help='log filters',
                        dest='log_filter')

    subparsers = parser.add_subparsers(help='commands')
    train_parser = subparsers.add_parser('train', help='starts training')
    train_parser.add_argument('--train', default='train', dest='task')
    train_parser.add_argument('--mux', action='store_true', default=False, dest='mux')

    dump_vocab_parser = subparsers.add_parser('dump-vocab',
                                              help='dumps the vocabulary into two tsv file')
    dump_vocab_parser.add_argument('--dump-vocab', default='dump-vocab', dest='task')

    dump_cosine_similarity_parser = subparsers.add_parser('dump-cosine-similarity',
                                                          help='dumps the cosine-similarity into two tsv file')
    dump_cosine_similarity_parser.add_argument('--dump-cosine-similarity',
                                               default='dump-cosine-similarity',
                                               dest='task')

    
    predict_parser = subparsers.add_parser('predict',
                                help='''starts a cli interface for running predictions 
                                in inputs with best model from last training run''')
    predict_parser.add_argument('--predict', default='predict', dest='task')
    predict_parser.add_argument('--show-plot', action='store_true', dest='show_plot')
    predict_parser.add_argument('--save-plot', action='store_true',  dest='save_plot')
    args = parser.parse_args()
    print(args)
    if args.log_filter:
        log.addFilter(CMDFilter(args.log_filter))

    ROOT_DIR = initialize_task(args.hpconfig)

    sys.path.append('.')
    print(sys.path)
    HPCONFIG = importlib.__import__(args.hpconfig.replace('.py', ''))
    config.HPCONFIG = HPCONFIG.CONFIG
    config.ROOT_DIR = ROOT_DIR
    config.NAME = SELF_NAME
    print('====================================')
    print(ROOT_DIR)
    print('====================================')
        
    if config.CONFIG.flush:
        log.info('flushing...')
        dataset = load_data(config, filename=config.HPCONFIG.dataset_path)
        pickle.dump(dataset, open('{}/{}__cache.pkl'.format(ROOT_DIR, SELF_NAME), 'wb'))
    else:
        dataset = pickle.load(open('{}/{}__cache.pkl'.format(ROOT_DIR, SELF_NAME), 'rb'))

        
    log.info('dataset size: {}'.format(len(dataset.trainset)))
    for i in range(10):
        log.info('random sample: {}'.format(pformat(random.choice(dataset.trainset))))

    #log.info('vocab: {}'.format(pformat(dataset.output_vocab.freq_dict)))
    ########################################################################################
    # load model snapshot data 
    ########################################################################################
    _batchop = partial(batchop, VOCAB=dataset.input_vocab, LABELS=dataset.output_vocab, config=config)
    train_feed     = DataFeed(SELF_NAME,
                              dataset.trainset,
                              batchop    = _batchop,
                              batch_size = config.CONFIG.batch_size)
    
    
    loss_ = partial(loss, loss_function=nn.NLLLoss())
    test_feed      = DataFeed(SELF_NAME, dataset.testset, batchop=_batchop, batch_size=config.CONFIG.batch_size)
    
    model =  Model(config, 'Model',
                   len(dataset.input_vocab),
                   len(dataset.output_vocab),
                   loss_function = loss_,
                   dataset = dataset,
                   train_feed = train_feed,
                   test_feed = test_feed,)
    
    print('**** the model', model)
    model.restore_checkpoint()
    
    if config.CONFIG.cuda:
        model = model.cuda()        
        if config.CONFIG.multi_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)


    
    if args.task == 'train':
        model.do_train()
        
    if args.task == 'dump-vocab':
        ids = torch.arange(model.vocab_size)
        embedding = model.__(model.embed(ids.cuda()), 'emb')
        
        dump_vocab_tsv(config,
                       dataset.input_vocab,
                       embedding.cpu().detach().numpy(),
                       config.ROOT_DIR + '/vocab.tsv')


    if args.task == 'dump-cosine-similarity':
        dump_cosine_similarity_tsv(config,
                   dataset.input_vocab,
                   model.embed.weight.data.cpu(),
                   config.ROOT_DIR + '/cosine_similarity.tsv')
        
        
    if args.task == 'predict':
        for i in range(10):
            try:
                output = model.do_predict()
            except:
                log.exception('#########3')
                pass
                
