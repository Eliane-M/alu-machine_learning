#!/usr/bin/env python3
'''
A class that represents a single neuron performing binary classification
'''


import numpy as np


class Neuron:
    '''
    Defines nx as the number of input features to the neuron
    Initializes the weights W and bias b to random values
    '''
    def __init__(self, nx):
        if isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.W = np.random.rand()
        self.b = 0
        self.A = 0