#!/usr/bin/env python3
'''
Defines a deep neural network performing binary classification
'''

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """
    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Parameters:
        - nx (int): The number of input features.
        - layers (list): A list representing the number of nodes in each layer of the network.
        
        Raises:
        - TypeError: If nx is not an integer or layers is not a list of positive integers.
        - ValueError: If nx is less than 1.
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            layer_size = layers[l - 1]
            prev_layer_size = nx if l == 1 else layers[l - 2]

            self.__weights[f"W{l}"] = np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            self.__weights[f"b{l}"] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X

        for l in range(1, self.__L + 1):
            Wl = self.__weights[f"W{l}"]
            bl = self.__weights[f"b{l}"]
            A_prev = self.__cache[f"A{l - 1}"]

            Zl = np.matmul(Wl, A_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))  # Sigmoid activation

            self.__cache[f"A{l}"] = Al

        return self.__cache[f"A{self.__L}"], self.__cache
