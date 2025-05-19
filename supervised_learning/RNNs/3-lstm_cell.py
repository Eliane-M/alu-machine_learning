#!/usr/bin/env python3
"""
Defines the class LSTMCell that represents an LSTM unit.
"""

import numpy as np


class LSTMCell:
    """
    Represents a LSTM unit.

    Class constructor:
        def __init__(self, i, h, o)

    Public instance attributes:
        Wf: forget gate weights
        bf: forget gate biases
        Wu: update gate weights
        bu: update gate biases
        Wc: intermediate cell state weights
        bc: intermediate cell state biases
        Wo: output gate weights
        bo: output gate biases
        Wy: output weights
        by: output biases

    Public instance methods:
        def forward(self, h_prev, c_prev, x_t):
            performs forward propagation for one time step
    """

    def __init__(self, i, h, o):
        """
        Class constructor.

        Parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs

        Creates public instance attributes:
            Wf: forget gate weights
            bf: forget gate biases
            Wu: update gate weights
            bu: update gate biases
            Wc: intermediate cell state weights
            bc: intermediate cell state biases
            Wo: output gate weights
            bo: output gate biases
            Wy: output weights
            by: output biases

        Weights should be initialized using random normal distribution.
        Weights will be used on the right side for matrix multiplication.
        Biases should be initialized as zeros.
        """
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """
        Performs the softmax function.

        Parameters:
            x: the value to perform softmax on to generate output of cell

        Returns:
            softmax of x
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def sigmoid(self, x):
        """
        Performs the sigmoid function.

        Parameters:
            x: the value to perform sigmoid on

        Returns:
            sigmoid of x
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev: Previous hidden state, numpy.ndarray of shape (m, h)
            c_prev: Previous cell state, numpy.ndarray of shape (m, h)
            x_t: Input data, numpy.ndarray of shape (m, i)

        Returns:
            h_next: Next hidden state
            c_next: Next cell state
            y: Output of the cell
        """
        # Concatenate h_prev and x_t for matrix multiplication
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        ft = self.sigmoid(np.dot(h_x, self.Wf) + self.bf)

        # Update gate
        ut = self.sigmoid(np.dot(h_x, self.Wu) + self.bu)

        # Intermediate cell state
        cct = np.tanh(np.dot(h_x, self.Wc) + self.bc)

        # Next cell state
        c_next = ft * c_prev + ut * cct

        # Output gate
        ot = self.sigmoid(np.dot(h_x, self.Wo) + self.bo)

        # Next hidden state
        h_next = ot * np.tanh(c_next)

        # Output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
