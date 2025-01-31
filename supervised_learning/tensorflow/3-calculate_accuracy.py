#!/usr/bin/env python3
'''
Calculates accuracy of a given prediction
'''


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    '''
    Calculates the accuracy of a given prediction
    '''
    correct_predictions = tf.equal(
        tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
