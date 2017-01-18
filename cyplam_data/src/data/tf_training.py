from __future__ import print_function

import tensorflow as tf
import numpy as np

import labeling
import analysis
import tf_cnn

from sklearn.utils import shuffle
from sklearn import model_selection


# Parameters
TRAINING_ITERS = 100000
BATCH_SIZE = 128
DISPLAY_STEP = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units


def read_datasets(dirnames, offset=0):
    filenames = labeling.get_filenames(dirnames)
    data, labels = labeling.read_datasets(filenames, offset=offset)
    frames = [analysis.read_frames(d.frame)[:,2:30,2:30].reshape(-1, 784) / 1024. for d in data]
    features, targets = labeling.label_data(frames, labels)
    features = features[targets >= 0]
    targets = targets[targets >= 0]
    t = np.zeros((len(targets), 10))
    for k in range(len(targets)):
        t[k][targets[k]] = 1
    targets = t
    return features.astype(np.float32), targets.astype(np.float32)


def run_training(dirnames):
    features, targets = read_datasets(dirnames, offset=0)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        features, targets, test_size=0.6, random_state=0)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    pred = tf_cnn.model(x, y, keep_prob)
    cost, optimizer = tf_cnn.training(pred, y)
    accuracy = tf_cnn.evaluation(pred, y)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * BATCH_SIZE < TRAINING_ITERS:
            # batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            batch_x, batch_y = shuffle(X_train, y_train, random_state=0, n_samples=BATCH_SIZE)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            if step % DISPLAY_STEP == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        # Save the variables to disk.
        save_path = saver.save(sess, "../../config/model.ckpt")
        print("Model saved in file: %s" % save_path)

        # Calculate accuracy for 256 mnist test images
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
        #                                   y: mnist.test.labels[:256],
        #                                   keep_prob: 1.}))
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.}))


if __name__ == '__main__':
    import os
    home = os.path.expanduser("~")
    dirnames = [os.path.join(home, './data/data_nov24/24112016_2v_1000')]
    run_training(dirnames)
