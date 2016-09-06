#!/usr/bin/env python
import rospy
from mashes_tachyon.msg import MsgCalibrate


class Calibrate():
    def __init__(self, parent=None):
        self.pub_calibrate = rospy.Publisher(
            '/tachyon/calibrate', MsgCalibrate, queue_size=10)
        self.msg_calibrate = MsgCalibrate()

    def run(self):
        self.msg_calibrate.calibrate = 1
        self.pub_calibrate.publish(self.msg_calibrate)

if __name__ == "__main__":
    calibrate = Calibrate()
    calibrate.run()
