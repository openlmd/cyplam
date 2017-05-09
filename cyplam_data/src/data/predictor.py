from __future__ import print_function
import tensorflow as tf
import numpy as np

import tf_cnn_linear


class PowerPredictor:
    def __init__(self,file_name):
        self._file_name = file_name
        self._x = tf.placeholder(tf.float32, [None, 784])
        self._y = tf.placeholder(tf.float32, [None, 1])
        self._keep_prob = tf.placeholder(tf.float32)
        self._pred = tf_cnn_linear.model(self._x, self._y, self._keep_prob)
        self._session = tf.Session()
        new_saver = tf.train.Saver()
        new_saver.restore(self._session, file_name)


    def run(self,image):
        image = np.asarray(np.reshape(image[2:-2,2:-2],(1,784)),np.float32)/1024
        y = np.ones((1,1))
        predicted = self._session.run(self._pred, feed_dict={self._x: image, self._y: y, self._keep_prob: 1.})
        return predicted[0][0]*100




if __name__ == '__main__':
    import os
    import rospkg
    from python_qt_binding import loadUi
    path = rospkg.RosPack().get_path('cyplam_data')
    loadUi(os.path.join(path, 'config/models', '2017_05_05_14_27_06model.ckpt'))
    pred = PowerPredictor(path)
    print(pred.run(np.ones((32,32))))
    print(pred.run(np.zeros((32,32))))
