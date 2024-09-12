#!/usr/bin/env python3
"""
This function concatenates two ndarray matrices
along a specific axis
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two ndarray matrices along
    a specific axis
    """
    np_cat = np.concatenate((mat1, mat2), axis=axis)
    return np_cat
