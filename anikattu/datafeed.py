from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import random

from anikattu.utilz import tqdm
from anikattu.debug import memory_consumed

from collections import Counter

class DataFeed(object):
    def __init__(self, name, dataset, batchop, batch_size=1, sort_key=None):
        self.name        = name
        self._offset     = 0
        self._size       = len(dataset)
        self._batch_size = batch_size
        self._batchop    = batchop
        self._batch_cache = {}
        self._exhausted_count = 0
        self.dataset  = dataset
        
        log.info('built Datafeed: {} with the following props:'.format(self.name))
        log.info(' size       : {}'.format(self.size))
        log.info(' batch_size : {}'.format(self.batch_size))
        log.info(' num_batch  : {}'.format(self.num_batch))
        
    @property
    def data(self): return self.dataset

    @property
    def data_dict(self): return self._data_dict
    
    @property
    def size(self): return self._size

    @property
    def batch_size(self): return self._batch_size

    @property
    def num_batch(self): return int(self.size/self.batch_size)
    
    @property
    def offset(self): return self._offset

    def __repr__(self): return 'DataFeed-{}:\n\t{}'.format(self.name, self.size)
    
    def batch(self, apply_batchop=True):
        self._offset += self.batch_size
        b = self.data [self.offset-self.batch_size : self.offset]

        if apply_batchop:
            return self._batchop(b)
        return b
    
    def next_batch(self, apply_batchop=True, **kwargs):
        try:
            if self.offset + self.batch_size > self.size:
                self._exhausted_count += 1
                self.reset_offset()
                
                log.debug('datafeed: {} over run - resetting offset to zero for {} time'
                          .format(self.name, self._exhausted_count))

            return self.batch(apply_batchop=apply_batchop)
        except SystemExit:
            exit(1)
        except KeyboardInterrupt:
            raise KeyboardInterrupt

        except:
            log.exception('batch failed')
            return self.next_batch(apply_batchop=apply_batchop)

    def nth_batch(self, n, apply_batchop=True):
        b =    self.data [n*self.batch_size : (n+1)*self.batch_size]
        if apply_batchop:
            return self._batchop(b)
        return b        

    def reset_offset(self):
        self._offset = 0


