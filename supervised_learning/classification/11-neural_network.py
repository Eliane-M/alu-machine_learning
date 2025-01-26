#!/usr/bin/env python3
'''
defines a neural network with one hidden layer performing binary classification
'''


import numpy as np


class NeuralNetwork:
    '''
    A neural network with one hidden layer performing binary classification
    '''
    def __init__(self, nx, nodes):
        '''
        Initializes the neural network with given input features, hidden nodes,
        and activation function
        Args:
            nx: Number of input features
            nodes: Number of nodes in the hidden layer
            activation: Activation function for the hidden layer
        '''
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases for the hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Initialize weights and biases for the output neuron
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    # Getter for b1
    @property
    def b1(self):
        return self.__b1

    # Getter for A1
    @property
    def A1(self):
        return self.__A1

    # Getter for W2
    @property
    def W2(self):
        return self.__W2

    # Getter for b2
    @property
    def b2(self):
        return self.__b2

    # Getter for A2
    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        '''
        Performs forward propagation through the neural network
        Args:
            X: Input features
        Returns:
            Output of the neural network
        '''
        # Calculate the input to the hidden layer
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Calculate the input for the output layer and apply the sigmoid activation function
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
    
    def cost(self, Y, A):
        '''
        Calculates the cost of the model using logistic regression
        '''
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
