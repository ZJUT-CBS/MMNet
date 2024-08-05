import numpy as np
import os


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        # Trainging
        self.learning_rate = 0.001
        self.weight_decay = 0.001
        self.num_folds = 10

        self.epochs = 300
        self.batch_size = 128
        self.batch_size_per_class = 40  # if equal sampling
        self.dropout_keep_prob = 0.8
        self.dropout_keep_prob_conv = 0.8

        self.evaluate_every = 50

        self.n_dim = 129  # frequency dimension
        self.n_time = 29  # time dimension
        self.n_channel = 3  # channel dimension
        self.n_class = 5  # Final output classes

        self.result = '.\data'
        self.model_pth = '.\model'
