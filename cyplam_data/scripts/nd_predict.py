#!/usr/bin/env python
import os
import rospy
import rospkg

# from cladplus_control.msg import MsgControl
# from cladplus_control.msg import MsgPower
from cyplam_data.msg import MsgPredictv
from cyplam_data.msg import MsgPredictp
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import random


class NdPredict():
    def __init__(self):
        rospy.init_node('prediction')
        rospy.Subscriber(
            '/tachyon/image', Image, self.cb_image, queue_size=1)
        self.pub_predictp = rospy.Publisher(
            '/predict/power', MsgPredictv, queue_size=10)
        self.pub_predictv = rospy.Publisher(
            '/predict/velocity', MsgPredictp, queue_size=10)

        self.bridge = CvBridge()

        self.msg_predictv = MsgPredictv()
        self.msg_predictp = MsgPredictp()

        rospy.spin()

    def cb_image(self, msg_image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg_image)
            time = msg_image.header.stamp
            self.image = frame
            v, p = self.predict(self.image)
            self.msg_predictv.value = v
            self.msg_predictv.header.stamp = time
            self.msg_predictp.value = p
            self.msg_predictp.header.stamp = time
            self.pub_predictv.publish(self.msg_predictv)
            self.pub_predictp.publish(self.msg_predictp)
        except CvBridgeError, e:
            print e

    def predict(self, image):
        #funcion equivalente a regresion de adrian
        return (random.randint(3, 8), random.randint(500, 1500))





if __name__ == '__main__':
    try:
        NdPredict()
    except rospy.ROSInterruptException:
        pass
