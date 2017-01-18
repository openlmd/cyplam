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
