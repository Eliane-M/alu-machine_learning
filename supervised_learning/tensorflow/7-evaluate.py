#!/usr/bin/env python3
'''
This function evaluates the output of the nueral network
'''


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Parameters:
    - X (numpy.ndarray): Input data to evaluate.
    - Y (numpy.ndarray): One-hot encoded labels for X.
    - save_path (str): Path to load the model from.

    Returns:
    - tuple: (predictions, accuracy, loss)
    """
    # Initialize variables to store results
    predictions = None
    accuracy_value = None
    loss_value = None

    # Create new session
    with tf.Session() as session:
        try:
            # Import meta graph and restore weights
            saver = tf.train.import_meta_graph(save_path + '.meta')
            saver.restore(session, save_path)

            # Get graph
            graph = tf.get_default_graph()

            # Get required tensors from collection
            y_pred = graph.get_collection('y_pred')[0]
            loss = graph.get_collection('loss')[0]
            accuracy = graph.get_collection('accuracy')[0]

            # Get input tensor (assuming it's named 'x')
            x = graph.get_collection('x')[0]
            # Get target tensor (assuming it's named 'y')
            y = graph.get_collection('y')[0]

            # Run evaluation
            predictions, accuracy_value, loss_value = session.run(
                [y_pred, accuracy, loss],
                feed_dict={
                    x: X,
                    y: Y
                }
            )

        except Exception as e:
            raise Exception(f"Error during model evaluation: {str(e)}")

    return predictions, accuracy_value, loss_value
