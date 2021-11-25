import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):

    dataset = 'news'
    window_size = 3
    dataset_path = '../dataset/data/text_lengthsorted.txt'
    max_samples = 10000
    freq_threshold = 100
    trainset_size = 1
    hidden_dim = 100
    embed_dim = 100

    kv_size = 1000
    input_channels = [1]
    output_channels = [hidden_dim]
    kernel_size = [3]
    
    LR = 0.01
    MOMENTUM=0.1
    ACTIVATION = 'softmax'
