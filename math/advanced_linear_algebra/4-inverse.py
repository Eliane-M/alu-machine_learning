#!/usr/bin/env python3
"""
This function calculates
the inverse matrix of a matrix
"""


def inverse(matrix):
    """
    Calculates the inverse matrix of a matrix
    """
    # check if the matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    # check if the matrix is square or not empty
    if len(set(len(row)
               for row in matrix)) != 1 or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    # if n == 1:
    #     return [[1]]

    def minor(mat, i, j):
        """Calculate the minor of matrix mat for element at (i, j)."""
        return [row[:j] + row[j + 1:] for row in (mat[:i] + mat[i + 1:])]

    def determinant(mat):
        """Calculate the determinant of a matrix."""
        if len(mat) == 1:
            return mat[0][0]
        if len(mat) == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        det = 0
        for j in range(len(mat)):
            det += ((-1) ** j) * mat[0][j] * determinant(minor(mat, 0, j))
        return det

    det = determinant(matrix)

    if det == 0:
        return None

    if n == 1:
        return [[1 / matrix[0][0]]]

    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            minor_det = determinant(minor(matrix, i, j))
            cofactor_row.append((-1) ** (i + j) * minor_det)
        cofactor_matrix.append(cofactor_row)

    adjugate_matrix = [[cofactor_matrix[j][i]
                        for j in range(n)] for i in range(n)]

    inverse_matrix = [[adjugate_matrix[i][j] /
                       det for j in range(n)] for i in range(n)]

    return inverse_matrix
