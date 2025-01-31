#!/usr/bin/env python3
'''
This function creates a layer and returns the tensor output of the layer
'''


import tensorflow as tf


def create_layer(prev, n, activation):
    '''
    Creates a layer with the given number of neurons and activation function.
    '''
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name="layer")
    return layer(prev)
