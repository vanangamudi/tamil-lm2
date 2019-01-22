import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    dataset = 'filmreviews'
    dataset_path = ('../dataset/filmreviews/reviews.subword_nmt.csv',
               '../dataset/filmreviews/ratings.csv')
    
    trainset_size = 1
    hidden_dim = 5
    embed_dim = 50
    num_layers = 1
    
    LR = 0.00001
    MOMENTUM=0.01
    ACTIVATION = 'softmax'

