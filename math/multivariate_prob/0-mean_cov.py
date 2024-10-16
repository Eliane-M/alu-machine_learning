#!/usr/bin/env python3
"""
function def mean_cov(X):
that calculates the mean and covariance of a data set
"""


import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean
    mean = np.mean(X, axis=0, keepdims=True)

    # Calculate the covariance matrix
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
