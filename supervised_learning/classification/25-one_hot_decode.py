#!/usr/bin/env python3
'''
converts a one-hot matrix into a vector of labels
'''

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
