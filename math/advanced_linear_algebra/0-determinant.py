#!/usr/bin/env python3
"""
Function that calculates the determinant of a matrix
"""


def determinant(matrix):
    """"
    determinant of a matrix
    """

    # check if the matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    # check if the matrix is a square matrix
    if len(set(len(row)
               for row in matrix)) != 1 or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # for a 0x0 matrix
    if matrix == [[]]:
        return 1

    # For a 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]

    # calculate the determinant for 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # initialize the determinant
    det = 0

    # iterate over the first row
    for i in range(len(matrix)):
        submatrix = [[row[j] for j in range(len(matrix)) if j != i]
                     for row in matrix[1:]]
        det += (-1) ** i * matrix[0][i] * determinant(submatrix)

    return det
