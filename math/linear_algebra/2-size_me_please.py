#!usr/bin/env python3

def matrix_shape(matrix):
    '''
        Calculates the shape of a matrix
    '''
    mat_shape = []
    while isinstance(matrix, list):
        mat_shape.append(len(matrix))
        matrix = matrix[0]
    return mat_shape

