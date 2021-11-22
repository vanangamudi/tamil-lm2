import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):

    dataset = 'news'
    window_size = 3
    dataset_path = '../../tharavu/cholloadai-2021.txt.tsv'
    max_samples = 100000000
    freq_threshold = 100
    trainset_size = 1
    hidden_dim = 100
    embed_dim = 300
    num_layers = 1
    
    LR = 0.01
    MOMENTUM=0.1
    ACTIVATION = 'softmax'
