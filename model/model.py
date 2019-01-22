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
        

        ########################################################################################
        #  Saving model weights
        ########################################################################################

        self.best_model = (1e-5, self.cpu().state_dict())

        
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
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
            metric = self.test_loss
            
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
                    self.log.info('loaded last best accuracy: {}'.format(self.best_model[0]))
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
            self.log.info('saving the last best model with accuracy {}...'
                          .format(self.best_model[0]))

            torch.save(self.best_model[1],
                       '{}/weights/{:0.4f}.{}'.format(self.config.ROOT_DIR,
                                                      self.best_model[0],
                                                      'pth'))
            
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
        self.accuracy   = Averager(self.config,
                                   filename = '{}.{}'.format(self.mfile_prefix,  'accuracy'))
        
        self.metrics = [self.train_loss, self.test_loss, self.accuracy]
        # optional metrics
        if getattr(self, 'f1score_function'):
            self.tp = Averager(self.config, filename = '{}.{}'.format(self.mfile_prefix,   'tp'))
            self.fp = Averager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'fp'))
            self.fn = Averager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'fn'))
            self.tn = Averager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'tn'))
            
            self.precision = Averager(self.config,
                                      filename = '{}.{}'.format(self.mfile_prefix,  'precision'))
            self.recall    = Averager(self.config,
                                      filename = '{}.{}'.format(self.mfile_prefix,  'recall'))
            self.f1score   = Averager(self.config,
                                      filename = '{}.{}'.format(self.mfile_prefix,  'f1score'))
          
            self.metrics += [self.tp, self.fp, self.fn, self.tn,
                             self.precision, self.recall, self.f1score]

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
                 accuracy_function=None,

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
        self.encode = nn.LSTM(self.embed_dim, self.hidden_dim,
                              bidirectional=True, num_layers=config.HPCONFIG.num_layers)
        
        self.classify = nn.Linear(2*self.hidden_dim, self.vocab_size)

        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        
        if accuracy_function :
            self.accuracy_function = accuracy_function
        else:
            self.accuracy_function = lambda *x, **xx: 1 / loss_function(*x, **xx)

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
                        
        self.best_model_criteria = self.accuracy

        if config.CONFIG.cuda:
            self.cuda()
        
    def restore_and_save(self):
        self.restore_checkpoint()
        self.save_best_model()
        
    def init_hidden(self, batch_size):
        ret = torch.zeros(2, batch_size, self.hidden_dim)
        if config.HPCONFIG().cuda: ret = ret.cuda()
        return Variable(ret)
    
    def forward(self, input_):
        ids, (seq,), _ = input_
        if seq.dim() == 1: seq = seq.unsqueeze(0)
            
        batch_size, seq_size = seq.size()
        seq_emb = F.tanh(self.embed(seq))
        seq_emb = seq_emb.transpose(1, 0)
        pad_mask = (seq > 0).float()
        
        states, cell_state = self.encode(seq_emb)
        
        logits = self.classify(states[-1])
        
        return F.log_softmax(logits, dim=-1)

   
    def do_train(self):
        for epoch in range(self.epochs):
            self.log.critical('memory consumed : {}'.format(memory_consumed()))            
            self.epoch = epoch
            if epoch % max(1, (self.checkpoint - 1)) == 0:
                #self.do_predict()
                if self.do_validate() == FLAGS.STOP_TRAINING:
                    self.log.info('loss trend suggests to stop training')
                    return
                           
            self.train()
            losses = []
            for j in tqdm(range(self.train_feed.num_batch), desc='Trainer.{}'.format(self.name())):
                self.optimizer.zero_grad()
                input_ = self.train_feed.next_batch()
                idxs, inputs, targets = input_

                output = self.forward(input_)
                loss   = self.loss_function(output, input_)
                #print(loss.data.cpu().numpy())
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
                idxs, inputs, targets = input_

                output   = self.forward(input_)
                loss     = self.loss_function(output, input_)
                accuracy = self.accuracy_function(output, input_)
                
                losses.append(loss)
                accuracies.append(accuracy)

            epoch_loss = torch.stack(losses).mean()
            epoch_accuracy = torch.stack(accuracies).mean()

            self.test_loss.append(epoch_loss.data.item())
            self.accuracy.append(epoch_accuracy.data.item())
                #print('====', self.test_loss, self.accuracy)

            self.log.info('= {} =loss:{}'.format(self.epoch, epoch_loss))
            self.log.info('- {} -accuracy:{}'.format(self.epoch, epoch_accuracy))
            
        if len(self.best_model_criteria) > 1 and self.best_model[0] < self.best_model_criteria[-1]:
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
    
    def do_predict(self, input_=None):
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

    
