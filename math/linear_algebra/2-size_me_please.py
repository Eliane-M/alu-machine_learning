#!/usr/bin/env python3
'''
    Given a list of lists (matrix), this function
    calculates and returns its shape.
'''

def matrix_shape(matrix):
    '''
        Calculates the shape of a matrix
    '''
    mat_shape = []
    while isinstance(matrix, list):
        mat_shape.append(len(matrix))
        matrix = matrix[0]
    return mat_shape
