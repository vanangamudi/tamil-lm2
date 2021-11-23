import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):

    dataset = 'news'
    
    dataset_path = '../../tharavu/cholloadai-2021.txt'
    max_samples = 1000000000
    trainset_size = 1
    hidden_dim = 100
    embed_dim = 100
    num_layers = 1
    
    LR = 0.001
    MOMENTUM=0.1
    ACTIVATION = 'softmax'
