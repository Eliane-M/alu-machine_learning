#!/usr/bin/env python3
'''
Defines a single neuron performing binary classification
'''


import numpy as np


class Neuron:
    '''
    A single neuron performing binary classification
    '''
    def __init__(self, nx):
        '''
        Initializes the neuron with given weights and bias
        Args:
            weights (numpy.ndarray): The weights of the neuron
            bias (float): The bias of the neuron
        '''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''
        Returns the weights of the neuron
        Returns:
            numpy.ndarray: The weights of the neuron
        '''
        return self.__W

    @property
    def b(self):
        '''
        Returns the bias of the neuron
        Returns:
            float: The bias of the neuron
        '''
        return self.__b

    @property
    def A(self):
        '''
        Returns the activation of the neuron
        Returns:
            float: The activation of the neuron
        '''
        return self.__A
