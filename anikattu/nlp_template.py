import os
import pdb
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')

from anikattu.logger import CMDFilter
import logging
from pprint import pprint, pformat

logging.basicConfig(format=config.FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import config

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.trainer import Trainer, Feeder, Predictor
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.utilz import tqdm, ListTable

from functools import partial

from collections import namedtuple, defaultdict
import itertools

from utilz import Sample
from utilz import PAD,  word_tokenize
from utilz import VOCAB, LABELS
from utilz import rotate

from anikattu.utilz import initialize_task
from anikattu.utilz import pad_seq
from anikattu.utilz import logger
from anikattu.vocab import Vocab
from anikattu.tokenstring import TokenString
from anikattu.utilz import LongVar, Var, init_hidden
import numpy as np

import re
import glob

SELF_NAME = os.path.basename(__file__).replace('.py', '')

def build_sample(raw_sample):
    pass

def prep_samples(dataset):
    ret = []
    vocabulary = defaultdict(int)
    labels = defaultdict(int)

    for i, sample in tqdm(enumerate(dataset)):
        try:
            sample = build_sample(sample)
            if not sample.label in LABELS:
                continue
            for token in sample.sentence:
                vocabulary[token] += 1
            labels[sample.label] += 1
            ret.append(sample)
        except KeyboardInterrupt:
            return
        except:
            log.exception('at id: {}'.format(i))

    return ret, vocabulary, labels


# ## Loss and accuracy function
def loss(output, batch, loss_function, *args, **kwargs):
    indices, (sentence, ), (label, ) = batch
    output, attn = output
    return loss_function(output, label)

def accuracy(output, batch, *args, **kwargs):
    indices, (sentence), (label, ) = batch
    output, attn = output
    return (output.max(dim=1)[1] == label).sum().float()/label.size(0)

def waccuracy(output, batch, *args, **kwargs):
    indices, (sentence, ), (label, ) = batch
    output, attn = output
    index = label
    src = Var(torch.ones(label.size()))
    
    acc_nomin = Var(torch.zeros(output.size(1)))
    acc_denom = Var(torch.ones(output.size(1)))

    acc_denom.scatter_add_(0, index, (label == label).float() )
    acc_nomin.scatter_add_(0, index, (label == output.max(1)[1]).float())

    accuracy = acc_nomin / acc_denom

    #pdb.set_trace()
    return accuracy.mean()

def f1score(output, input_, *args, **kwargs):

    indices, (seq, ) , (target,) = input_
    output, attn = output
    batch_size, class_size  = output.size()

    tp, tn, fp, fn = Var([0]), Var([0]), Var([0]), Var([0])
    p, r, f1 = Var([0]), Var([0]), Var([0])

    i = output
    t = target
    i = i.max(dim=1)[1]
    log.debug('output:{}'.format(pformat(i)))
    log.debug('target:{}'.format(pformat(t)))
    i_ = i
    t_ = t
    tp_ = ( i_ * t_ ).sum().float()
    fp_ = ( i_ > t_ ).sum().float()
    fn_ = ( i_ < t_ ).sum().float()

    i_ = i == 0
    t_ = t == 0
    tn_ = ( i_ * t_ ).sum().float()

    tp += tp_
    tn += tn_
    fp += fp_
    fn += fn_

    log.debug('tp_: {}\n fp_:{} \n fn_: {}\n tn_: {}'.format(tp_, fp_, fn_, tn_))


    if tp_.data.item() > 0:
        p_ = tp_ / (tp_ + fp_)
        r_ = tp_ / (tp_ + fn_)
        f1 += 2 * p_ * r_/ (p_ + r_)
        p += p_
        r += r_

    return (tp, fn, fp, tn), (p), (r), (f1)

def repr_function(output, batch, VOCAB, LABELS):
    indices, (sentence,), (label,) = batch
    
    results = []
    output, attn = output
    output = output.max(1)[1]
    output = output.cpu().numpy()
    for idx, c, a, o in zip(indices, sentence, label, output):
        #if not int(a) == int(o) and LABELS[a] == 'Y':
        if True:
            c = ' '.join([VOCAB[i] for i in c])
            a = ' '.join([LABELS[a]])
            o = ' '.join([LABELS[o]])
            
            results.append([ c, a, o, str(a == o) ])
        
    return results


def test_repr_function(output, batch, VOCAB, LABELS):
    indices, (sentence,), (label,) = batch
    
    results = []
    score, attn = output
    attn = attn.transpose(0, 1).squeeze(2)
    score, output = score.max(1)
    score = score.exp()
    for idx, c, a, o, s, at in zip(indices, sentence, label, output, score, attn):
        results.append([idx,
                        ' '.join([VOCAB[i] for i in c]),
                        ' '.join([LABELS[o]]),
                        '{:0.4f}'.format(s),
                        ','.join(['{:0.4f}'.format(i) for i in at.tolist()]),
                        repr([VOCAB[i] for i in c])
        ])
        
    return results

def batchop(datapoints, VOCAB, LABELS, *args, **kwargs):
    indices = [d.id for d in datapoints]
    sentence = []
    label = []

    for d in datapoints:
        sentence.append([VOCAB[w] for w in d.sentence] + [VOCAB['EOS']])
        #sentence.append([VOCAB[w] for w in d.sentence])
        label.append(LABELS[d.label])

    sentence    = LongVar(pad_seq(sentence))
    label   = LongVar(label)

    batch = indices, (sentence, ), (label, )
    return batch

class Base(nn.Module):
    def __init__(self, config, name):
        super(Base, self).__init__()
        self._name = name
        self.log = logging.getLogger(self._name)
        size_log_name = '{}.{}'.format(self._name, 'size')
        self.log.info('constructing logger: {}'.format(size_log_name))
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.info('size_log')
        self.log.setLevel(logging.INFO)
        self.size_log.setLevel(logging.INFO)
        self.print_instance = 0
        
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
            if self.print_instance or print_instance:
                self.size_log.debug(tensor)

            
        return tensor

    def name(self, n):
        return '{}.{}'.format(self._name, n)


class BiLSTMModel(Base):
    pass

    
def experiment(config, ROOT_DIR, model, VOCAB, LABELS, datapoints=[[], [], []], eons=1000, epochs=20, checkpoint=1):
    try:
        name = SELF_NAME
        _batchop = partial(batchop, VOCAB=VOCAB, LABELS=LABELS)
        train_feed     = DataFeed(name, datapoints[0], batchop=_batchop, batch_size=config.HPCONFIG.batch_size)
        test_feed      = DataFeed(name, datapoints[1], batchop=_batchop, batch_size=config.HPCONFIG.batch_size)
        predictor_feed = DataFeed(name, datapoints[2], batchop=_batchop, batch_size=1)

        max_freq = max( LABELS.freq_dict[i] for i in LABELS.index2word  )
        loss_weight = [ 1/ ( LABELS.freq_dict[i]/ max_freq) for i in LABELS.index2word ]
        print(list((l, w) for l, w in zip(LABELS.index2word, loss_weight)))
        loss_weight = Var(loss_weight)
        
        loss_ = partial(loss, loss_function=nn.NLLLoss(loss_weight))
        trainer = Trainer(name=name,
                          model=model,
                          optimizer = optim.SGD(model.parameters(),
                                                lr=config.HPCONFIG.OPTIM.lr,
                                                momentum=config.HPCONFIG.OPTIM.momentum),
                          loss_function=loss_, accuracy_function=waccuracy, f1score_function=f1score,
                          checkpoint=checkpoint, epochs=epochs,
                          directory = ROOT_DIR,
                          feeder = Feeder(train_feed, test_feed))

        predictor = Predictor(model=model.clone(), feed=predictor_feed,
                              repr_function=partial(test_repr_function, VOCAB=VOCAB, LABELS=LABELS))
        
        for e in range(eons):

            if not trainer.train():
                raise Exception

            predictor.model.load_state_dict(trainer.best_model[1])
            
            dump = open('{}/results/eon_{}.csv'.format(ROOT_DIR, e), 'w')
            log.info('on {}th eon'.format(e))
            results = ListTable()
            for ri in tqdm(range(predictor_feed.num_batch)):
                output, _results = predictor.predict(ri)
                results.extend(_results)
            dump.write(repr(results))
            dump.close()
            
            


    except KeyboardInterrupt:
        return locals()
    except :
        log.exception('####################')
        return locals()
    
import sys
import pickle
if __name__ == '__main__':

    if sys.argv[1]:
        log.addFilter(CMDFilter(sys.argv[1]))

    ROOT_DIR = initialize_task(SELF_NAME)

    print('====================================')
    print(ROOT_DIR)
    print('====================================')
    
    if config.CONFIG.flush or 'flush' in sys.argv:
        log.info('flushing...')
        dataset = []
        with open('../dataset/dataset.csv') as f:
            for line in tqdm(f.readlines()):
                line = line.split('|')
                dataset.append(
                    Sample(
                        line[0], line[1], line[2]
                    )
                )
        dataset, vocabulary, labels =  prep_samples(dataset)
        pivot = int( config.CONFIG.split_ratio * len(dataset) )
        trainset, testset = dataset[:pivot], dataset[pivot:]
        pickle.dump([trainset, testset, dict(vocabulary), dict(labels)], open('{}__cache.pkl'.format(SELF_NAME), 'wb'))
    else:
        trainset, testset, _vocabulary, _labels = pickle.load(open('{}__cache.pkl'.format(SELF_NAME), 'rb'))
        vocabulary = defaultdict(int); labels = defaultdict(int)
        vocabulary.update(_vocabulary), labels.update(_labels)
        
    log.info('trainset size: {}'.format(len(trainset)))
    log.info('trainset[:10]: {}'.format(pformat(trainset[0])))

    pprint(labels)
    """
    log.info('vocabulary: {}'.format(
        pformat(
            sorted(
                vocabulary.items(), key=lambda x: x[1], reverse=True)
        )))
    """
    

    log.info(pformat(labels))
    VOCAB  = Vocab(vocabulary, VOCAB)
    LABELS = Vocab(labels, tokens=LABELS)
    pprint(LABELS.index2word)


    try:
        model =  BiLSTMModel(config, 'macnet', len(VOCAB),  len(LABELS))
        if config.CONFIG.cuda:  model = model.cuda()
        model.load_state_dict(torch.load('{}/weights/{}.{}'.format(ROOT_DIR, SELF_NAME, 'pth')))
        log.info('loaded the old image for the model')
    except:
        log.exception('failed to load the model')

    model.eval()
    print('**** the model', model, model.training)
    
    if 'train' in sys.argv:
        model.train()
        train_set = sorted(trainset, key=lambda x: -len(x.sentence))
        test_set  = sorted(testset, key=lambda x: -len(x.sentence))
        exp_image = experiment(config, ROOT_DIR,  model, VOCAB, LABELS, datapoints=[train_set, train_set + test_set, train_set + test_set])
        
    if 'predict' in sys.argv:
        print('=========== PREDICTION ==============')
        model.eval()
        count = 0
        while True:
            count += 1
            sentence = []
            input_string = word_tokenize(input('?').lower())
            sentence.append([VOCAB[w] for w in input_string] + [VOCAB['EOS']])
            dummy_label = LongVar([0])
            sentence = LongVar(sentence)
            input_ = [0], (sentence,), (0, )
            output, attn = model(input_)

            print(LABELS[output.max(1)[1]])

            if 'show_plot' in sys.argv or 'save_plot' in sys.argv:
                nwords = len(input_string)

                from matplotlib import pyplot as plt
                plt.figure(figsize=(20,10))
                plt.bar(range(nwords+1), attn.squeeze().data.cpu().numpy())
                plt.title('{}\n{}'.format(output.exp().tolist(), LABELS[output.max(1)[1]]))
                plt.xticks(range(nwords), input_string, rotation='vertical')
                if 'show_plot' in sys.argv:
                    plt.show()
                if 'save_plot' in sys.argv:
                    plt.savefig('{}.png'.format(count))
                plt.close()

            print('Done')
                
    if 'service' in sys.argv:
        model.eval()
        from flask import Flask,request,jsonify
        from flask_cors import CORS
        app = Flask(__name__)
        CORS(app)

        @app.route('/ade-genentech',methods=['POST'])
        def _predict():
           print(' requests incoming..')
           sentence = []
           try:
               input_string = word_tokenize(request.json["text"].lower())
               sentence.append([VOCAB[w] for w in input_string] + [VOCAB['EOS']])
               dummy_label = LongVar([0])
               sentence = LongVar(sentence)
               input_ = [0], (sentence,), (0, )
               output, attn = model(input_)
               #print(LABELS[output.max(1)[1]], attn)
               nwords = len(input_string)
               return jsonify({
                   "result": {
                       'sentence': input_string,
                       'attn': ['{:0.4f}'.format(i) for i in attn.squeeze().data.cpu().numpy().tolist()[:-1]],
                       'probs': ['{:0.4f}'.format(i) for i in output.exp().squeeze().data.cpu().numpy().tolist()],
                       'label': LABELS[output.max(1)[1].squeeze().data.cpu().numpy()]
                   }
               })
           
           except Exception as e:
               print(e)
               return jsonify({"result":"model failed"})

        print('model running on port:5010')
        app.run(host='0.0.0.0',port=5010)
