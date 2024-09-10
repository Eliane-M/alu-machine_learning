#!/usr/bin/env python3
"""
    This program demonstrates the concept of matrix transposition.
    It takes a 2D list (matrix) as input,
    and returns its transpose.

    The transpose of a matrix is obtained by
    swapping its rows with its columns.
"""


def matrix_transpose(matrix):
    """
    This function takes a 2D list (matrix)
    as input and returns its transpose.
    """
    transpose = [[0 for _ in range(len(matrix))]
                 for _ in range(len(matrix[0]))]

    for j in range(len(matrix)):
        for i in range(len(matrix[0])):
            transpose[i][j] = matrix[j][i]
    return transpose