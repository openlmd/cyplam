


import numpy as np
import tensorflow as tf


n_input = 784
n_classes = 1
dropout = 0.75
learning_rate = 0.0001

# Create some wrappers for simplicity
def conv2d3(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 5, 1], padding='SAME')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x23d(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 1, 1],
                        strides=[1, 2, 2, 1, 1], padding='SAME')



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def model(x, y, dropout):



    x = tf.reshape(x, shape=[-1, 28, 28, 1])


    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)



    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])


    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

    W_fc2 = weight_variable([1024, n_classes])
    b_fc2 = bias_variable([n_classes])

    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y

def loss(pred, y):
    cost = tf.reduce_mean(tf.pow(pred - y, 2))
    return cost


def training(pred, y):
    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.pow(pred-y, 2))
    #cross_entropy = tf.reduce_mean(
    #    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

    print(tf.trainable_variables())
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    return cross_entropy, optimizer



def evaluation(pred, y):
    # Evaluate model
    correct_pred = tf.less(tf.abs(tf.reduce_mean(pred,axis=1)-tf.reduce_mean(y,axis=1)),0.5)#tf.equal(tf.max(pred, 1), tf.max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


