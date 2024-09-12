#!/usr/bin/env python3
"""
Function slices a matrix
along a specific axis
"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along a specific axis.
    """
    np_slice = [slice(None)] * matrix.ndim

    for axis, slice_info in axes.items():
        np_slice[axis] = slice(*slice_info)

    return matrix[tuple(np_slice)]
