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
        self.d = d

    def pdf(self, x):
        """
        Calculates the probability density function (pdf) of the multivariate normal
        distribution at the given point x.
        """
        # Validate input x
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        # Calculate PDF
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        x_centered = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv_cov), x_centered)
        coefficient = 1 / np.sqrt((2 * np.pi) ** self.d * det_cov)

        return float(coefficient * np.exp(exponent))
