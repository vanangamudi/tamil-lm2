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
from anikattu.utilz import tqdm, ListTable, dump_vocab_tsv, conv2d_output_size
from .model import Base

class Conv2dEmbedding(Base):
    def __init__(self, config, name,
                 input_dim, embed_dim,
                 conv2d_out_channels,
                 conv2d_kernel_size,
                 conv2d_padding):
        
        super().__init__(config, name)
        self.input_dim  = input_dim
        self.embed_dim = embed_dim

        
        self._embed = []
        for i in range(self.embed_dim):
            mod = nn.Embedding(self.input_dim, self.embed_dim)
            self._embed.append(mod)
            self.add_module('{}_{}'.format(name, i), mod)

        
        self.convolve = nn.Conv2d(in_channels  = 1,
                                  out_channels = conv2d_out_channels,
                                  kernel_size  = conv2d_kernel_size,
                                  padding      = conv2d_padding)

        W, H = conv2d_output_size(self.embed_dim, self.embed_dim, 3, 1)
        self.output_dim =  W * H * self.convolve.out_channels
        
    def forward(self, input_):
        batch_size = input_.size(0)
        emb = []
        for i in self._embed:
            emb.append(i(input_))

        emb = self.__(torch.stack(emb), 'emb')


        emb = self.__( emb.permute(0, 1, 2), 'emb')
        emb = self.__( emb.view(-1, self.embed_dim, self.embed_dim), 'emb')

        emb = self.__(self.convolve(emb.unsqueeze(1)), 'convolved_emb')
        emb = self.__(emb.view(batch_size, self.output_dim),
                      'convolved_emb')
        
        return emb

    
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
        self.vocab_size = input_vocab_size
        self.hidden_dim = config.HPCONFIG.hidden_dim
        self.embed_dim = config.HPCONFIG.embed_dim

        #self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        #self.project = nn.Linear(self.embed_dim, self.vocab_size)


        self.embed = Conv2dEmbedding(self.config, 'conv2dembed',
                                     self.vocab_size, self.embed_dim,
                                     conv2d_out_channels=5,
                                     conv2d_kernel_size=3,
                                     conv2d_padding=1)

        print('output_dim', self.embed.output_dim)
        self.project = nn.Linear(self.embed.output_dim, self.vocab_size)
        
        self.loss_function = loss_function if loss_function else nn.NLLLoss()
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

        
        self.optimizer = optimizer if optimizer else optim.SGD(self.parameters(),
                                                               lr=1, momentum=0.1)
        self.optimizer = optimizer if optimizer else optim.Adam(self.parameters())        
        if config.CONFIG.cuda:
            self.cuda()
        
    def restore_and_save(self):
        self.restore_checkpoint()
        self.save_best_model()
        
    def forward(self, word):
        emb = self.__( self.embed(word), 'emb' )
        logits = self.__( self.project(emb), 'logis')
        return F.log_softmax(logits, dim=-1)
   
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
                idxs, word, targets = input_
                output = self.__(self.forward(word), 'output')
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

    def build_cache_for_train2(self):
        self.batch_cache = []
        for j in tqdm(range(self.train_feed.num_batch), desc='building cache'):
            input_ = self.train_feed.next_batch()
            self.batch_cache.append(input_)
    
    def do_train2(self):
        if not hasattr(self, 'batch_cache'):
            self.build_cache_for_train2()
        
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
            for input_ in tqdm(self.batch_cache, desc='Trainer.{}'.format(self.name())):
                self.optimizer.zero_grad()
                idxs, word, targets = input_
                output = self.__(self.forward(word), 'output')
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
                idxs, word, targets = input_
                output = self.__(self.forward(word), 'output')
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

