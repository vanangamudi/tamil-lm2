from config import CONFIG
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import random
import mmap
from anikattu.utilz import tqdm
from anikattu.debug import memory_consumed
from anikattu.vocab import Vocab

from collections import Counter
import random

def split_dataset(dataset, ratio=0.8):
    pivot = ratio * len(dataset)
    return dataset[:pivot], dataset[pivot:]


class Dataset:
    def __init__(self, name, dataset_path, line_pos, delim, vocab):

        self.name = name
        log.info('building dataset: {}'.format(name))

        self.dataest_path = dataset_path
        dataset_file = open(dataset_path)
        self.dataset_file = mmap.mmap(dataset_file.fileno(),
                                      length=0,
                                      access=mmap.ACCESS_READ)
        self.line_pos = line_pos[:1000]
        self.delim = delim

        
        self.input_vocab = self.output_vocab = vocab

    def __len__(self):
        return len(self.line_pos)
        
    def __getitem__( self, key ) :
        if isinstance( key, slice ) :
            inputs, outputs = [], []
            self.dataset_file.seek( self.line_pos[key.start] )
            for i in range(key.start, key.stop):
                line = self.dataset_file.readline()
                
                input_, output = line.split(self.delim)
                input_, output = int(input_), int(output)
                inputs.append(input_)
                outputs.append(output)
            return inputs, outputs
        
        elif isinstance( key, int ) :
            self.dataset_file.seek( self.line_pos[key] )
            line = self.dataset_file.readline()
            input_, output = line.split(self.delim)
            input_, output = int(input_), int(output)

            return input_, output
            
        else:
            raise TypeError("Invalid argument type. {}".format(key))

    
    def __del__(self):
        self.dataset_file.close()
