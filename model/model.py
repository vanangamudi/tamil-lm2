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

class Base(nn.Module):
    def __init__(self, config, name):
        super(Base, self).__init__()
        self.config = config
        self._name = name
        self.log = logging.getLogger(self._name)
        size_log_name = '{}.{}'.format(self._name, 'size')
        self.log.info('constructing logger: {}'.format(size_log_name))
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.info('size_log')
        self.log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.size_log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.print_instance = 0
        
        
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            size = tensor.size()
            self.size_log.debug('{} -> {}'.format(name, size))
            if self.print_instance or print_instance:
                self.size_log.debug(tensor)

            
        return tensor

    def name(self, n=''):
        if n:
            return '{}.{}'.format(self._name, n)
        else:
            return self._name

        
    def loss_trend(self, metric = None, total_count=10):
        if not metric:
            metric = self.best_model_criteria
            
        if len(metric) > 4:
            losses = metric[-4:]
            count = 0
            for l, r in zip(losses, losses[1:]):
                if l < r:
                    count += 1
                    
            if count > total_count:
                return FLAGS.STOP_TRAINING

        return FLAGS.CONTINUE_TRAINING


    def restore_checkpoint(self):
        try:
            self.snapshot_path = '{}/weights/{}.{}'.format(self.config.ROOT_DIR, self.name(), 'pth')
            self.load_state_dict(torch.load(self.snapshot_path))
            log.info('loaded the old image for the model from :{}'.format(self.snapshot_path))

                
            try:
                f = '{}/{}_best_model_accuracy.txt'.format(self.config.ROOT_DIR, self.name())
                if os.path.isfile(f):
                    self.best_model = (float(open(f).read().strip()), self.cpu().state_dict())
                    self.log.info('loaded last best metric: {}'.format(self.best_model[0]))
            except:
                log.exception('no last best model')
            
            if self.config.CONFIG.cuda:
                self.cuda()    
        except:
            
            log.exception('failed to load the model  from :{}'.format(self.snapshot_path))


            
    def save_best_model(self):
        with open('{}/{}_best_model_accuracy.txt'
                  .format(self.config.ROOT_DIR, self.name()), 'w') as f:
            f.write(str(self.best_model[0]))

        if self.save_model_weights:
            self.log.info('saving the last best model with metric {}...'
                          .format(self.best_model[0]))
            """
            torch.save(self.best_model[1],
                       '{}/weights/{:0.4f}.{}'.format(self.config.ROOT_DIR,
                                                      self.best_model[0],
            'pth'))
            """
            
            torch.save(self.best_model[1],
                       '{}/weights/{}.{}'.format(self.config.ROOT_DIR,
                                                 self.name(),
                                                 'pth'))


    def __build_stats__(self):
        ########################################################################################
        #  Saving model weights
        ########################################################################################
        
        # necessary metrics
        self.mfile_prefix = '{}/results/metrics/{}'.format(self.config.ROOT_DIR, self.name())
        self.train_loss  = Averager(self.config,
                                    filename = '{}.{}'.format(self.mfile_prefix,   'train_loss'))
        
        self.test_loss  = Averager(self.config,
                                   filename = '{}.{}'.format(self.mfile_prefix,   'test_loss'))
        
        self.metrics = [self.train_loss, self.test_loss]

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

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.encode = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        
        self.classify = nn.Linear(self.hidden_dim, self.vocab_size)

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
        
    def init_hidden(self, batch_size):
        hidden  = Variable(torch.zeros(batch_size, self.hidden_dim))
        cell  = Variable(torch.zeros(batch_size, self.hidden_dim))
        if config.CONFIG.cuda:
            hidden  = hidden.cuda()
            cell    = cell.cuda()
            
        return hidden, cell

    
    def forward(self, prev_output, state):
        prev_output_emb = self.__( self.embed(prev_output), 'prev_output_emb' )
        hidden, cell_state = self.encode(prev_output_emb, state) 
        logits = self.classify(hidden)        
        return F.log_softmax(logits, dim=-1), (hidden, cell_state)
    
   
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
                idxs, seq, targets = input_

                seq_size, batch_size = seq.size()
                pad_mask = (seq > 0).float()

                loss = 0
                outputs = []
                output = self.__(seq[0], 'output')
                state = self.__(self.init_hidden(batch_size), 'init_hidden')
                for index in range(seq_size - 1):
                    output, state = self.__(self.forward(output, state), 'output, state')
                    loss   += self.loss_function(output, targets[index+1])
                    output = self.__(output.max(1)[1], 'output')
                    outputs.append(output)
                    
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
                idxs, seq, targets = input_

                seq_size, batch_size = seq.size()
                pad_mask = (seq > 0).float()

                loss = 0
                outputs = []
                output = self.__(seq[0], 'output')
                state = self.__(self.init_hidden(batch_size), 'init_hidden')
                for index in range(seq_size - 1):
                    output, state = self.__(self.forward(output, state), 'output, state')
                    loss   += self.loss_function(output, targets[index+1])
                    output = self.__(output.max(1)[1], 'output')
                    outputs.append(output)
                    
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
            input_ = self.test_feed.nth_batch(
                random.randint(0, self.test_feed.size),
                1
            )

        print(input_)
        ids, seq, label = input_
        seq_size, batch_size = seq.size()
                    
        outputs = []
        output = self.__(seq[0], 'output')
        state = self.__(self.init_hidden(batch_size), 'init_hidden')
        for index in range(max_len - 1):
            output, state = self.__(self.forward(output, state), 'output, state')
            output = self.__(output.max(1)[1], 'output')
            outputs.append(output)

        outputs = torch.stack(outputs)
        print(output.size())

        print(self.dataset.input_vocab[seq[0]])
        print(' '.join([self.dataset.input_vocab[i.data[0]] for i in outputs]).replace('@@ ', ''))
        
        return True

    
