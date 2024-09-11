#!/usr/bin/env python3
"""
Show how to multiply
two matrices
"""


def mat_mul(mat1, mat2):
    """
    Multiply two matrices
    """

    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result
