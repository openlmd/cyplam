#!/usr/bin/env python
import os
import sys
import rospy
import rospkg
import rosparam
import numpy as np
from std_msgs.msg import String, Header
# from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
# from nav_msgs.msg import Path

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore


path = rospkg.RosPack().get_path('cyplam_robviz')


class QtData(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        loadUi(os.path.join(path, 'resources', 'data.ui'), self)

        self.btnConnect.clicked.connect(self.btnConnectClickced)
        self.btnJob.clicked.connect(self.btnJobClicked)
        self.btnPredict.clicked.connect(self.btnPredictClicked)

    def btnConnectClickced(self):
        print 'Connect'
        print 'User:', self.txtUser.text()
        print 'Password:', self.txtPassword.text()

    def btnJobClicked(self):
        print 'Job:', self.txtName.text()

    def btnPredictClicked(self):
        print 'Base Material:', self.txtBaseMaterial.text()
        print 'Powder Material:', self.txtPowderMaterial.text()
        print 'Predict'


if __name__ == "__main__":
    rospy.init_node('data_panel')

    app = QtGui.QApplication(sys.argv)
    qt_data = QtData()
    qt_data.show()
    app.exec_()
