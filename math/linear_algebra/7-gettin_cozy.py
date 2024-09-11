#!/usr/bin/env python3
"""
Concetenates two matrices
along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.
    """
    row1, column1 = len(mat1), len(mat1[0])
    row2, column2 = len(mat2), len(mat2[0])

    # Check dimensions based on the axis
    if axis == 0 and column1 != column2:
        return None  # For vertical concatenation, column count must match
    if axis == 1 and row1 != row2:
        return None  # For horizontal concatenation, row count must match

    # Create a deep copy to avoid mutating original matrices
    result = [row[:] for row in mat1]

    # Vertical concatenation (along rows)
    if axis == 0:
        result.extend([row[:] for row in mat2])
    
    # Horizontal concatenation (along columns)
    elif axis == 1:
        for i in range(row1):
            result[i].extend(mat2[i])

    return result
