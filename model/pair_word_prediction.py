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
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from functools import partial


import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from anikattu.utilz import Var, LongVar, init_hidden, Averager, FLAGS, tqdm
from anikattu.debug import memory_consumed
from anikattu.utilz import tqdm, ListTable, dump_vocab_tsv
from .model import Base

class Model(Base):
    def __init__(self, config, name,
                 input_vocab_size,
                 output_vocab_size,
    
                 # feeds
                 dataset,
                 train_feed,
                 test_feed,

                 # loss function
                 loss_function,

                 f1score_function=None,
                 save_model_weights=True,
                 epochs = 1000,
                 checkpoint = 1,
                 early_stopping = True,

                 # optimizer
                 optimizer = None,):
        
        
        super(Model, self).__init__(config, name)
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = config.HPCONFIG.hidden_dim
        self.embed_dim = config.HPCONFIG.embed_dim

        self.embed = nn.Embedding(self.input_vocab_size, self.embed_dim)
        self.encode = nn.LSTM(self.embed_dim, self.embed_dim)
        self.project = nn.Linear(2 * self.embed_dim, self.output_vocab_size)

        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        
        self.optimizer = optimizer if optimizer else optim.SGD(self.parameters(),
                                                               lr=0.01, momentum=0.1)
        
        self.f1score_function = f1score_function
        
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping

        self.dataset = dataset
        self.train_feed = train_feed
        self.test_feed = test_feed
        
        self.save_model_weights = save_model_weights

        self.__build_stats__()
                        
        self.best_model_criteria = self.train_loss

        if config.CONFIG.cuda:
            self.cuda()
        
    def restore_and_save(self):
        self.restore_checkpoint()
        self.save_best_model()
        
    def forward(self, pair):
        w1_state, w2_state = self.encode_pair(pair)

        state = torch.cat([w1_state, w2_state], dim=-1)
        
        logits = self.__( self.project(state), 'logis')
        return F.log_softmax(logits, dim=-1)

    
    def encode_pair(self, pair):
        w1, w2 = pair
        w1_emb = self.__( self.embed(w1).transpose(1, 0), 'w1_emb' )
        w2_emb = self.__( self.embed(w2).transpose(1, 0), 'w2_emb' )

        w1_states, _ = self.__(self.encode(w1_emb), 'w1_states')
        w2_states, _ = self.__(self.encode(w2_emb), 'w2_states')

        return w1_states[-1], w2_states[-1]
        
    
    def do_train(self):
        for epoch in range(self.epochs):
            self.log.critical('memory consumed : {}'.format(memory_consumed()))            
            self.epoch = epoch
            if epoch and epoch % max(1, (self.checkpoint - 1)) == 0:
                #self.do_predict()
                if self.do_validate() == FLAGS.STOP_TRAINING:
                    self.log.info('loss trend suggests to stop training')
                    return
                           
            self.train()
            losses = []
            for j in tqdm(range(self.train_feed.num_batch), desc='Trainer.{}'.format(self.name())):
                self.optimizer.zero_grad()
                input_ = self.train_feed.next_batch()
                idxs, pair, targets = input_
                output = self.__(self.forward(pair), 'output')
                loss   = self.loss_function(output, targets)
                    
                losses.append(loss)
                loss.backward()
                self.optimizer.step()

            epoch_loss = torch.stack(losses).mean()
            self.train_loss.append(epoch_loss.data.item())

            self.log.info('-- {} -- loss: {}\n'.format(epoch, epoch_loss))
            for m in self.metrics:
                m.write_to_file()

        return True

    def do_validate(self):
        self.eval()
        if self.test_feed.num_batch > 0:
            losses, accuracies = [], []
            for j in tqdm(range(self.test_feed.num_batch), desc='Tester.{}'.format(self.name())):
                input_ = self.test_feed.next_batch()
                idxs, pair, targets = input_
                output = self.__(self.forward(pair), 'output')
                loss   = self.loss_function(output, targets)
                
                losses.append(loss)

            epoch_loss = torch.stack(losses).mean()

            self.test_loss.append(epoch_loss.data.item())

            self.log.info('= {} =loss:{}'.format(self.epoch, epoch_loss))
            
        if len(self.best_model_criteria) > 1:
            if self.best_model_criteria[-2] > self.best_model_criteria[-1]:
                self.log.info('beat best ..')
                self.best_model = (self.best_model_criteria[-1],
                                   self.cpu().state_dict())                             
                
                self.save_best_model()
                #self.dump_vocab_tsv()
            
                if self.config.CONFIG.cuda:
                    self.cuda()
        
        for m in self.metrics:
            m.write_to_file()
            
        if self.early_stopping:
            return self.loss_trend()
    
    def do_predict(self, input_=None, max_len=10):
        if not input_:
            input_ = self.train_feed.nth_batch(
                random.randint(0, self.train_feed.size),
                1
            )
        
        output = self.forward(input_)
        output = output.max(1)[1].long()
        print(output.size())

        ids, (sequence, ), (label) = input_
        print(' '.join([self.dataset.input_vocab[i.data[0]] for i in sequence[0]]).replace('@@ ', ''))
        print(self.dataset.output_vocab[output.data[0]],
              ' ==? ',
              self.dataset.output_vocab[label.data[0]] )
        
        return True

    def dump_vocab_tsv(self, filepath=None):

        embeddings = {}
        self.eval()
        self.train_feed.reset_offset()
        for j in tqdm(range(self.train_feed.size), desc='dump emb'):
            input_ = self.train_feed.next_batch(batch_size=1)
            idxs, pair, targets = input_
            w1_state, w2_state = self.encode_pair(pair)

            sample = self.dataset.trainset_dict[idxs[0]] #since batch size = 1 idxs[0] works
            w1, w2 = sample.pair
            w1_text = ' '.join(w1).replace('@@ ', '')
            w2_text = ' '.join(w1).replace('@@ ', '')

            embeddings[w1_text] = w1_state.tolist()[0]
            embeddings[w2_text] = w2_state.tolist()[0]


        if not filepath:
            filepath = self.config.ROOT_DIR + '/vocab.tsv'
            
        vector_filepath = filepath.replace('.tsv', '.vector.tsv')
        token_filepath  = filepath.replace('.tsv', '.token.tsv')
        
        vector_file = open(vector_filepath, 'w')
        token_file  = open(token_filepath,  'w')    

        for word, vector in tqdm(embeddings.items(), desc='writing to file'):
            vector_file.write('\t'.join([str(v) for v in vector]) + '\n')
            token_file.write(word + '\n')
            
        vector_file.close()
        token_file.close()
        
