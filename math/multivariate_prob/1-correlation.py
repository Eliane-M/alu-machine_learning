#!/usr/bin/env python3
"""
function def correlation(C):
that calculates a correlation matrix
"""


import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Calculate standard deviations
    std_devs = np.sqrt(np.diag(C))

    # Prevent division by zero in case of zero variance
    if np.any(std_devs == 0):
        raise ValueError("Standard deviations cannot be zero")

    # Calculate the correlation matrix
    correlation_matrix = C / np.outer(std_devs, std_devs)

    return correlation_matrix
