#!/usr/bin/env python3
"""
This function adds matrices
element wise
"""


import numpy as np
def add_matrices2D(mat1, mat2):
    """
    Add matrices element-wise
    """
    # if len(mat1) != len(mat2):
    #     return None
    
    if np.shape(mat1) == np.shape(mat2):
        sum = np.add(mat1, mat2)
        return sum
    else:
        return None


