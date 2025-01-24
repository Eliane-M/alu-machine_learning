#!/usr/bin/env python3
'''
Defines a deep neural network performing binary classification
'''


import pickle
from matplotlib import pyplot as plt
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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        
        Y: numpy.ndarray with shape (1, m) - correct labels for the input data
        A: numpy.ndarray with shape (1, m) - activated output of the neuron for each example
        
        Returns:
            The cost of the model
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        
        X: numpy.ndarray with shape (nx, m) containing the input data
        Y: numpy.ndarray with shape (1, m) containing the correct labels
        
        Returns:
            A tuple containing:
                - The prediction: a numpy.ndarray with shape (1, m)
                  where the values are 1 if the output is >= 0.5, and 0 otherwise.
                - The cost of the network.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Y: numpy.ndarray with shape (1, m) containing the correct labels
        cache: dictionary containing all intermediary values of the network
        alpha: learning rate

        Updates the private attribute __weights.
        """
        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y

        for l in range(self.__L, 0, -1):
            A_prev = cache[f"A{l - 1}"]
            Wl = self.__weights[f"W{l}"]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Update weights and biases
            self.__weights[f"W{l}"] -= alpha * dW
            self.__weights[f"b{l}"] -= alpha * db

            if l > 1:  # Skip calculating dZ for the first layer
                dZ = np.matmul(Wl.T, dZ) * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        X: numpy.ndarray with shape (nx, m) containing the input data
        Y: numpy.ndarray with shape (1, m) containing the correct labels
        iterations: number of iterations to train over
        alpha: learning rate

        Returns:
        The evaluation of the training data after iterations of training.
        """
        # Input validation
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations):
            # Forward propagation
            A, cache = self.forward_prop(X)

            # Gradient descent
            self.gradient_descent(Y, cache, alpha)

            # Cost calculation and verbose output
            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
                print(f"Cost after {i} iterations: {cost}")

        # Final cost calculation and verbose output
        cost = self.cost(Y, A)
        if verbose:
            print(f"Cost after {iterations} iterations: {cost}")
        costs.append(cost)

        # Graph plotting
        if graph:
            plt.plot(range(0, iterations + 1, step), costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # Evaluate the final model
        return self.evaluate(X, Y)
    
    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        
        Parameters:
        - filename: The file to which the object should be saved.
        
        If the filename does not have the extension .pkl, it will be added.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object from a file.
        
        Parameters:
        - filename: The file from which the object should be loaded.
        
        Returns:
        - The loaded DeepNeuralNetwork object, or None if filename doesn't exist.
        """
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Error: The file {filename} does not exist.")
            return None
