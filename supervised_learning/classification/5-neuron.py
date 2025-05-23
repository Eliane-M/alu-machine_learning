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

    def forward_prop(self, X):
        '''
        Calculates the forward propagation of the neuron
        Args:
            X (numpy.ndarray): The input data
        Returns:
            float: The activation of the neuron
        '''
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        '''
        Calculates the cost of the model using logistic regression
        '''
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        '''
        Evaluate's the model's predictions
        '''
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        # Convert probability to binary predictions
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''
        Performs gradient descent to update the weights and bias of the neuron
        '''
        m = X.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        # Update weights and bias
        self.__W -= alpha * dW
        self.__b -= alpha * db
