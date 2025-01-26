#!/usr/bin/env python3
'''
converts a one-hot matrix into a vector of labels
'''


import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Parameters:
    Y (numpy.ndarray): A vector of numeric class labels with shape (m,)
    classes (int): The maximum number of classes found in Y

    Returns:
    numpy.ndarray: A one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
