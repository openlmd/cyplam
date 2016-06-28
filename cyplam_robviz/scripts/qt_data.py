#!/usr/bin/env python
import os
import sys
import yaml
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
    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)
        loadUi(os.path.join(path, 'resources', 'data.ui'), self)
        self.parent = parent
        self.btnConnect.clicked.connect(self.btnConnectClickced)
        self.btnJob.clicked.connect(self.btnJobClicked)
        self.btnPredict.clicked.connect(self.btnPredictClicked)
        home = os.path.expanduser('~')
        self.dirdata = os.path.join(home, 'bag_data')
        # self.setMaterialparam(rospy.get_param('/material'))

    def btnConnectClickced(self):
        self.save_access()

    def save_access(self):
        yalm = os.path.join(self.dirdata, 'access.yalm')
        print yalm
        # yalm = '/home/panadeiro/bag_data/job0001/access.yalm'
        # self.dirdata = os.path.join(self.dirdata, '/access.yalm')
        # print self.derdata
        accessdata = {'IP': '172.20.0.204', 'user': str(self.txtUser.text()), 'password': str(self.txtPassword.text())}
        with open(yalm, 'w') as outfile:
            outfile.write(yaml.dump(accessdata, default_flow_style=True))

    def btnJobClicked(self):
        print 'Job:', self.txtJobName.text()
        home = os.path.expanduser('~')
        self.dirdata = os.path.join(home, 'bag_data/%s' % self.txtJobName.text())
        if not os.path.exists(self.dirdata):
            os.mkdir(self.dirdata)
        if self.parent:
            self.parent.workdir = self.dirdata

    def btnPredictClicked(self):
        print 'Base Material:', self.txtBaseMaterial.text()
        print 'Powder Material:', self.txtPowderMaterial.text()
        param = self.getMaterialparam()
        rospy.set_param('/material', param)

    def setMaterialparam(self, params):
        print params
        self.txtBaseMaterial.setText(params['base'])
        self.txtPowderMaterial.setText(params['mat_powder'])

    def getMaterialparam(self):
        params = {'base': self.txtBaseMaterial.text(),
                  'mat_powder': self.txtPowderMaterial.text()}
        return params


if __name__ == "__main__":
    rospy.init_node('data_panel')

    app = QtGui.QApplication(sys.argv)
    qt_data = QtData(parent=None)
    qt_data.show()
    app.exec_()
