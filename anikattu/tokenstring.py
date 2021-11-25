import re

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class TokenString(list):

    def __init__(self, s, tokenize=None):
        if isinstance(s, list):
            assert tokenize != None, 'tokenize cannot be none for list input'
            self.raw_string = ''.join(s)
            self.tokenize = tokenize
            self.tokenized_string = s
        else:
            if tokenize == None:
                tokenize = str.split

            self.raw_string = s
            self.tokenize = tokenize
            self.tokenized_string = self.tokenize(s)

    @property
    def raw(self):
        return self.raw_string
    
    def __equals__(self, other):
        if isinstance(other, TokenString):
            return self.tokenized_string == other.tokenized_string
        elif isinstance(other, str):
            return self.raw_string == other

    def __contains__(self, subspan):
        return self.index(subspan) != None

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.tokenized_string[key]
        else:
            s = ' '.join(self.tokenized_string[key])
        return TokenString(s, self.tokenize)

    def __setitem__(self, key, value):
        if isinstance(value, TokenString):
            self.tokenized_string[key] = value.tokenized_string
        elif isinstance(value, list):
            self.tokenized_string[key] = value

    def __delitem__(self, key):
        del self.tokenized_string[key]
    
    def __repr__(self):
        return 'TokenString({})'.format(self.tokenized_string)

    def __len__(self):
        return len(self.tokenized_string)
    
    def __iter__(self):
        return self.tokenized_string.__iter__()

    def __add__(self, other):
        assert self.tokenize == other.tokenize, 'tokenstring created from different tokenizers are not supported yet'
        return TokenString(other.tokenized_string + self.tokenized_string, self.tokenize)
    
    def index(self, subspan, start_at=0, stop_at=None):
        watchdog = 0
        if not stop_at:
            stop_at = len(self)
        window = self.tokenized_string[start_at: stop_at]
        if isinstance(subspan, str):
            subspan = self.tokenize(subspan)
        
        if isinstance(subspan, TokenString) or isinstance(subspan, list):
            if not subspan[0] in window:
                log.debug('subspan not found returning None')
                return
 
            start_idx = window.index(subspan[0])
            while start_idx < len(window):
                log.debug('found subspan[0] at {}'.format(start_idx))
                offset = 0
                while start_idx + offset < len(window) and offset < len(subspan) and window[start_idx+offset] == subspan[offset]:
                    log.debug('{} == {}'.format(window[start_idx+offset], subspan[offset]))
                    log.debug('{}, {}'.format(start_idx, offset))
                    offset += 1
                    watchdog += 1
                    if watchdog > len(window):
                        log.error(pformat(locals()))
                        return
                    
                if offset == len(subspan):
                    log.debug('subspan found : {}, {}'.format(start_at+start_idx,
                                                              start_at+start_idx+offset))
                    return (start_at+start_idx, start_at+start_idx+offset)
                
                if not subspan[0] in window[start_idx+offset:]:
                    log.debug(locals())
                    log.debug('subspan not found after start_idx: {}'.format(start_idx+1))
                    log.debug(pformat(window[start_idx+offset:]))
                    return

                start_idx += offset
                start_idx += window[start_idx:].index(subspan[0])

    
    def indices(self, item):
        _indices = []
        index = self.index(item)
        offset = 0
        while index != None:
            _indices.append(index)
            offset = index[1] + 1
            index = self.index(item, start_at=offset)

        return _indices

    def replace(self, key, value, all=False):
        if all:
            for i in self.indices(key):
                self.tokenized_string[slice(*i)] = value

        else:
            i = self.index(key)
            if i:
                self.tokenized_string[slice(*i)]  = value   


    def delete_whitespace(self):
        self.tokenized_string = [i for i in self.tokenized_string
                                 if not i.isspace()]

        return self


import sys
from .tokenizer import word_tokenize
if __name__ == '__main__':

    t = TokenString(sys.argv[1], word_tokenize).delete_whitespace()
    s = TokenString(sys.argv[2], word_tokenize).delete_whitespace()

    print(s in t)
