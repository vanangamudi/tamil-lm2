import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):

    dataset = 'news'
    window_size = 2
    dataset_path = '../dataset/lm_lengthsorted_bpe30K.txt'
    max_samples = 1000000
    trainset_size = 1
    hidden_dim = 100
    embed_dim = 300
    num_layers = 1
    
    LR = 0.001
    MOMENTUM=0.1
    ACTIVATION = 'softmax'
