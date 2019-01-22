import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):

    dataset = 'news'
    
    labels={
        'tamilnadu': 3115,
        'india': 2263,
        'cinema': 1256,
        'sports': 1057,
        'world': 712,
        'politics': 702,
        #'special-news': 302,
        #'science-technology': 234,
        #'crime': 198,
        #'finance': 186,
        #'education-employement': 79,
        #'health': 30,
        #'agriculture': 21,
        #'districts': 17,
        #'head-news': 14,
        #'election': 10,
        #'others': 3,
    }
    
    dataset_path = '../dataset/lm_lengthsorted_bpe.txt'
    trainset_size = 1
    hidden_dim = 300
    embed_dim = 300
    num_layers = 1
    
    LR = 0.001
    MOMENTUM=0.1
    ACTIVATION = 'softmax'
