#!/usr/bin/env python3
'''
Neural-network based module using tensorflow
'''


import tensorflow as tf


def create_placeholders(nx, classes):
    '''
    Creates two placeholders, x and y, for the neural network.
    '''
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")
    return x, y
