'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np

from data import labeling
from data import analysis

from sklearn.utils import shuffle
from sklearn import model_selection


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


dirnames = ['/home/jorge/data/data_set23/20160923_2v_oven']
features, targets = read_datasets(dirnames, offset=0)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features, targets, test_size=0.6, random_state=0)


# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

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


# Create model
def conv_net(x, weights, biases, dropout):
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
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

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

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x, batch_y = shuffle(X_train, y_train, random_state=0, n_samples=batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Save the variables to disk.
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)

    # Calculate accuracy for 256 mnist test images
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
    #                                   y: mnist.test.labels[:256],
    #                                   keep_prob: 1.}))
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.}))



dirnames = ['/home/jorge/data/data_set23/20160922_1p',
            '/home/jorge/data/data_set23/20160923_2v',
            '/home/jorge/data/data_set23/20160923_3t',
            '/home/jorge/data/data_set23/20160923_4c',
            '/home/jorge/data/data_set23/20160923_5f',
            '/home/jorge/data/data_set23/20160923_6c',
            '/home/jorge/data/data_set23/20160923_1p_oven',
            '/home/jorge/data/data_set23/20160923_2v_oven',
            '/home/jorge/data/data_set23/20160923_3t_oven',
            '/home/jorge/data/data_set23/20160923_4c_oven',
            '/home/jorge/data/data_set23/20160923_5f_oven',
            '/home/jorge/data/data_set23/20160923_6c_oven']


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "model.ckpt")
    print("Model restored.")
    # Do some work with the model
    filenames = labeling.get_filenames(dirnames)
    for n, filename in enumerate(filenames):
        data, labels = labeling.read_datasets([filename], offset=0)
        for k, dat in enumerate(data):
            frames = analysis.read_frames(dat.frame)[:,2:30,2:30].reshape(-1, 784) / 1024.
            predicted = sess.run(tf.argmax(pred, 1), feed_dict={x: frames, keep_prob: 1.})
            score = float(np.sum(predicted))/len(predicted)
            sums = [np.sum(predicted == l) for l in range(5)]
            print('D', n, 't', k, 'score', score, 'labels', sums)
