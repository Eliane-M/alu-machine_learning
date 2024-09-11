#!/usr/bin/env python3
"""
A programm that will add two arrays
"""


def add_arrays(arr1, arr2):
    """
    This function adds two arrays together
    """
    if len(arr1) != len(arr2):
        return None

    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
