import numpy as np
import tensorflow as tf

import labeling
import analysis


# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def model(x, y, dropout):
    """Defines and constructs the CNN model."""

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    pred = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return pred


def loss(pred, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    return cost


def training(pred, y):
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    return cost, optimizer


def evaluation(pred, y):
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def do_evaluation(dirnames):
    # Network Parameters
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    pred = model(x, y, keep_prob)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "../../config/model.ckpt")
        print "Model restored."
        # Do some work with the model
        filenames = labeling.get_filenames(dirnames)
        for n, filename in enumerate(filenames):
            data, labels = labeling.read_datasets([filename], offset=0)
            for k, dat in enumerate(data):
                frames = analysis.read_frames(dat.frame)[:,2:30,2:30].reshape(-1, 784) / 1024.
                predicted = sess.run(tf.argmax(pred, 1), feed_dict={x: frames, keep_prob: 1.})
                score = float(np.sum(predicted))/len(predicted)
                sums = [np.sum(predicted == l) for l in range(5)]
                print 'D', n, 't', k, 'score', score, 'labels', sums


if __name__ == '__main__':
    import os
    home = os.path.expanduser("~")
    # dirnames = [os.path.join(home, './data/data_nov24/24112016_1v_900'),
    #             os.path.join(home, './data/data_nov24/24112016_2v_1000'),
    #             os.path.join(home, './data/data_nov24/24112016_3p_900'),
    #             os.path.join(home, './data/data_nov24/24112016_4p_1000')]
    dirnames = [os.path.join(home, './data/data_dec01/01122016_velocity_ramp_04'),
                os.path.join(home, './data/data_dec01/01122016_power_ramp_02'),
                os.path.join(home, './data/data_dec02/01122016_solape01_20'),
                os.path.join(home, './data/data_dec02/01122016_solape01_20_control'),
                os.path.join(home, './data/data_dec02/02122016_solape03_30_1300')]
    do_evaluation(dirnames)
