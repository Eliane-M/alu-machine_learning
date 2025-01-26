#!/usr/bin/env python3
'''
Defines a single neuron performing binary classification
'''


from matplotlib import pyplot as plt
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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron.

        Parameters:
        X (numpy.ndarray): Input data of shape (nx, m).
        Y (numpy.ndarray): Correct labels of shape (1, m).
        iterations (int): The number of iterations to train over.
        alpha (float): The learning rate.
        verbose (bool): If True, print the cost every `step` iterations.
        graph (bool): If True, graph the cost every `step` iterations.
        step (int): The step at which to display or graph cost.

        Returns:
        tuple: The evaluation of the training data after training.
        """
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
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                costs.append((i, cost))
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            x, y = zip(*costs)
            plt.plot(x, y, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
