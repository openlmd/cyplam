# import cv2
import numpy as np
from random import randint
from matplotlib import pyplot as plt

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

from data import labeling
from data import analysis

from sklearn.utils import shuffle
from sklearn import cross_validation


def read_datasets(dirnames, offset=0):
    filenames = labeling.get_filenames(dirnames)
    data, labels = labeling.read_datasets(filenames, offset=offset)
    frames = [analysis.read_frames(d.frame)[:,2:30,2:30].reshape(-1, 784) / 1024. for d in data]
    features, targets = labeling.label_data(frames, labels)
    return features, targets


def label_data(features, targets):
    features = features[targets >= 0]
    targets = targets[targets >= 0]
    t = np.zeros((len(targets), 10))
    for k in range(len(targets)):
        t[k][targets[k]] = 1
    targets = t
    return features.astype(np.float32), targets.astype(np.float32)


dirnames = ['/home/jorge/data/data_set23/20160923_2v_oven']
features, targets = read_datasets(dirnames, offset=0)
features, targets = label_data(features, targets)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    features, targets, test_size=0.6, random_state=0)

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Model
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# Train our model
for i in range(1000):
    batch_xs, batch_ys = shuffle(X_train, y_train, random_state=0, n_samples=100)
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluationg our model:
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print "Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print "Accuracy: ", sess.run(accuracy, feed_dict={x: X_test, y_: y_test})

# 1: Using our model to classify a random MNIST image from the original test set:
#num = randint(0, mnist.test.images.shape[0])
#img = mnist.test.images[num]
num = randint(0, features.shape[0])
img = features[num]

classification = sess.run(y, feed_dict={x: [img]})
# Uncomment this part if you want to plot the classified image.
plt.imshow(img.reshape(28, 28))
plt.show()

print 'Neural Network predicted', np.argmax(classification[0])
print 'Real label is:', np.argmax(targets[num])


dirnames = ['/home/jorge/data/data_set23/20160923_2v_oven']
features, targets = read_datasets(dirnames, offset=0)
data, labels = label_data(features, targets)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print "Accuracy: ", sess.run(accuracy, feed_dict={x: data, y_: labels})
predicted = sess.run(tf.argmax(y, 1), feed_dict={x: data})
print 'Predicted:', predicted



# # CNN model
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='SAME')
#
#
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
# # First convolutional layer
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])  # number of features
#
# x_image = tf.reshape(x, [-1, 28, 28, 1])  # image shape
#
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# # Second convolutional layer
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# # Densely connected layer
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# # Readout layer
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#
# # Training and evaluation CNN
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# sess = tf.Session()
# # sess.run(tf.initialize_local_variables())
# sess.run(tf.initialize_all_variables())
# # for i in range(20000):
# for i in range(1000):
#     batch_xs, batch_ys = shuffle(X_train, y_train, random_state=0, n_samples=50)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#     if i % 100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={
#             x: batch_xs, y_: batch_ys, keep_prob: 1.0})
#         print("step %d, training accuracy %g" % (i, train_accuracy))
#     train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
#
# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: X_test, y_: y_test, keep_prob: 1.0}))
