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

        self.L = len(layers)  # Number of layers in the network
        self.cache = {}  # Dictionary to store intermediate values
        self.weights = {}  # Dictionary to store weights and biases

        # Initialize weights and biases using the He et al. method
        for l in range(1, self.L + 1):
            layer_input_size = nx if l == 1 else layers[l - 2]
            self.weights[f"W{l}"] = (np.random.randn(layers[l - 1], layer_input_size)
                                       * np.sqrt(2 / layer_input_size))
            self.weights[f"b{l}"] = np.zeros((layers[l - 1], 1))
