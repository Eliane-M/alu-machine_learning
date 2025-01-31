#!/usr/bin/env python3
'''
This function creates a forward propagation graph for the neural network
'''


import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    '''
    Creates a forward propagation graph for a neural network
    Args:
        x: Input tensor
        layer_sizes: List of integers representing the number of neurons in each layer
        activations: List of activation functions for each layer
    Returns:
        A tensor representing the output of the neural network
    '''
    layer = x
    for i in range(len(layer_sizes)):
        activation = activations[i] if activations[i] is not None else None
        layer = create_layer(layer, layer_sizes[i], activation)
    return layer
