import logging

import re

class CMDFilter(logging.Filter):
    def __init__(self, pattern):
        self.must_pattern, self.must_not_pattern = pattern.split(' ## ')
        
    def filter(self, record):
        return (re.search(self.must_pattern, str(record)) and
                not re.search(self.must_not_pattern, str(record)))
