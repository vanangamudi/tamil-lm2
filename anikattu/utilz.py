from pprint import pprint, pformat

import os
import shutil
import pickle

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn
from torch.autograd import Variable

from collections import namedtuple, defaultdict

"""
    Local Utilities, Helper Functions

"""
def mkdir_if_exist_not(name):
    if not os.path.isdir(name):
        return os.mkdir(name)
    
def initialize_task(hpconfig = 'hpconfig.py', prefix='run00'):
    log.info('loading hyperparameters from {}'.format(hpconfig))
    root_dir = hpconfig.replace('.py', '') + '__' + hash_file(hpconfig)[-6:]
    mkdir_if_exist_not(prefix)
    root_dir = '{}/{}'.format(prefix, root_dir)
    mkdir_if_exist_not(root_dir)
    mkdir_if_exist_not('{}/results'.format(root_dir))
    mkdir_if_exist_not('{}/results/metrics'.format(root_dir))
    mkdir_if_exist_not('{}/weights'.format(root_dir))
    mkdir_if_exist_not('{}/plots'.format(root_dir))

    shutil.copy(hpconfig, root_dir)
    shutil.copy('config.py', root_dir)

    return root_dir

"""
Logging utils
"""
def logger(func, dlevel=logging.INFO):
    def wrapper(*args, **kwargs):
        level = log.getEffectiveLevel()
        log.setLevel(level)
        ret = func(*args, **kwargs)
        log.setLevel(level)
        return ret
    
    return wrapper


from pprint import pprint, pformat
from tqdm import tqdm as _tqdm

def tqdm(a, *args, **kwargs):
    return _tqdm(a, ncols=100,  *args, **kwargs) # if config.CONFIG.tqdm else a


def squeeze(lol):
    """
    List of lists to List

    Args:
        lol : List of lists

    Returns:
       List 

    """
    return [ i for l in lol for i in l ]

"""
    util functions to enable pretty print on namedtuple

"""
def _namedtuple_repr_(self):
    return pformat(self.___asdict())

def ___asdict(self):
    d = self._asdict()
    for k, v in d.items():
        if hasattr(v, '_asdict'):
            d[k] = ___asdict(v)

    return dict(d)


"""
# Batching utils   
"""
import numpy as np
def seq_maxlen(seqs):
    return max([len(seq) for seq in seqs])

PAD = 0
def pad_seq(seqs, maxlen=0, PAD=PAD):
    def pad_seq_(seq):
        return seq[:maxlen] + [PAD]*(maxlen-len(seq))

    if len(seqs) == 0:
        return seqs
    
    if type(seqs[0]) == type([]):
        maxlen = maxlen if maxlen else seq_maxlen(seqs)
        seqs = [ pad_seq_(seq) for seq in seqs ]
    else:
        seqs = pad_seq_(seqs)
        
    return seqs


class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
    the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
    IPython Notebook. 
    Taken from http://calebmadrigal.com/display-list-as-table-in-ipython-notebook/"""
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

    def __repr__(self):
        lines = []
        for i in self:
            lines.append('|'.join(i))
        log.debug('number of lines: {}'.format(len(lines)))
        return '\n'.join(lines + ['\n'])

"""
torch utils
"""
def are_weights_same(model1, model2):
    m1dict = model1.state_dict()
    m2dict = model2.state_dict()
    
    if m1dict.keys() != m2dict.keys():
        log.error('models don\'t match')
        log.error(pformat(m1dict.keys()))
        log.error(pformat(m2dict.keys()))
        return False
    
    for p in m1dict.keys():
        ne = m1dict[p].data.ne(m2dict[p].data)
        if ne.sum() > 0:
            print('===== {} ===='.format(p))
            print(ne.cpu().numpy())
            print('sum = ', ne.sum().cpu().numpy())
    
            return False
        
    return True

def LongVar(config, array, requires_grad=False):
    return Var(config, array, requires_grad).long()

def Var(config, array, requires_grad=False):
    ret =  Variable(torch.Tensor(array), requires_grad=requires_grad)
    if config.CONFIG.cuda:
        ret = ret.cuda()

    return ret

def init_hidden(config, batch_size, cell):
    layers = 1
    if isinstance(cell, (nn.LSTM, nn.GRU)):
        layers = cell.num_layers
        if cell.bidirectional:
            layers = layers * 2

    if isinstance(cell, (nn.LSTM, nn.LSTMCell)):
        hidden  = Variable(torch.zeros(layers, batch_size, cell.hidden_size))
        context = Variable(torch.zeros(layers, batch_size, cell.hidden_size))
    
        if config.CONFIG.cuda:
            hidden  = hidden.cuda()
            context = context.cuda()
        return hidden, context

    if isinstance(cell, (nn.GRU, nn.GRUCell)):
        hidden  = Variable(torch.zeros(layers, batch_size, cell.hidden_size))
        if config.CONFIG.cuda:
            hidden  = hidden.cuda()
        return hidden

class FLAGS:
    CONTINUE_TRAINING = 0
    STOP_TRAINING = 1
    
class Averager(list):
    def __init__(self, config, filename=None, ylim=None, *args, **kwargs):
        super(Averager, self).__init__(*args, **kwargs)
        self.config = config
        self.filename = filename
        self.ylim = ylim
        if filename:
            try:
                f = '{}.pkl'.format(filename)
                if os.path.isfile(f):
                    log.debug('loading {}'.format(f))
                    self.extend(pickle.load(open(f, 'rb')))
            except:
                open(filename, 'w').close()

    @property
    def avg(self):
        if len(self):
            return sum(self)/len(self)
        else:
            return 0

    def __str__(self):
        if len(self) > 0:
            #return 'min/max/avg/latest: {:0.5f}/{:0.5f}/{:0.5f}/{:0.5f}'.format(min(self), max(self), self.avg, self[-1])
            return '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(min(self), max(self), self.avg, self[-1])
        
        return '<empty>'

    def append(self, a):
        super(Averager, self).append(a)
            
    def empty(self):
        del self[:]

    def write_to_file(self):
        
        if self.filename:
            if self.config.CONFIG.plot_metrics:
                import matplotlib.pyplot as plt
                plt.plot(self)
                plt.title(os.path.basename(self.filename), fontsize=20)
                plt.xlabel('epoch')
                if self.ylim:
                    plt.ylim(*self.ylim)

                plt.savefig('{}.{}'.format(self.filename, 'png'))
                plt.close()

            pickle.dump(list(self), open('{}.pkl'.format(self.filename), 'wb'))
            with open(self.filename, 'a') as f:
                f.write(self.__str__() + '\n')
                f.flush()

    

class EpochAverager(Averager):
    def __init__(self, config, filename=None, *args, **kwargs):
        super(EpochAverager, self).__init__(config, filename, *args, **kwargs)
        self.config = config
        self.epoch_cache = Averager(config, filename, *args, *kwargs)

    def cache(self, a):
        self.epoch_cache.append(a)

    def clear_cache(self):
        super(EpochAverager, self).append(self.epoch_cache.avg)
        self.epoch_cache.empty();
                

# Python program to find SHA256 hash string of a file
#https://www.quickprogrammingtips.com/python/how-to-calculate-sha256-hash-of-a-file-in-python.html
import hashlib

def hash_file(filename):
    sha256_hash = hashlib.sha256()
    with open(filename,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()


