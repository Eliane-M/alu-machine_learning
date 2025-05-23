#!/usr/bin/env python3
'''
Function that builds, trains and saves
a neural network classifier
'''


import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
        X_train, Y_train, X_valid, Y_valid, layer_sizes,
        activations, alpha, iterations, save_path="/tmp/model.ckpt"
):
    '''
    Builds, trains and saves a neural network classifier

    :return: the path where the model was saved
    '''
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        '''
        Initialize variables and train the model
        '''
        sess.run(init)
        for i in range(iterations + 1):
            '''
            Run training operation; calculate
            training and validation loss & accuracy
            '''
            train_loss, train_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train}
            )
            valid_loss, valid_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid}
            )

            if i % 100 == 0 or i == iterations:
                '''
                Print training and validation loss & accuracy
                '''
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_loss}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_loss}")
                print(f"\tValidation Accuracy: {valid_acc}")

            if i < iterations:
                '''
                Run training operation
                '''
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        return saver.save(sess, save_path)
