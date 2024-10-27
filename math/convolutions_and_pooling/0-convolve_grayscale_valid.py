#!/usr/bin/env python3
"""a function that performs a valid convolution on grayscale img"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """a function that performs a valid convolution on grayscale img"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = h - kh + 1
    pw = w - kw + 1
    convolved = np.zeros((m, ph, pw))
    for i in range(ph):
        for j in range(pw):
            image = images[:, i:(i + kh), j:(j + kw)]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
