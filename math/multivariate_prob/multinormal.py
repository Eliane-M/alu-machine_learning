#!/usr/bin/env python3
"""
class MultiNormal
that represents a Multivariate Normal distribution
"""


import numpy as np


class MultiNormal():
    """
    class that a multivariate normal distribution
    """
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a 2D numpy.ndarray')

        # Validate data shape
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean and covariance matrix
        # Shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        # Shape (d, d)
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
