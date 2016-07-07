#!/usr/bin/env python
import os
import sys
import yaml
import rospy
import rospkg

import numpy as np

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore


path = rospkg.RosPack().get_path('cyplam_robviz')


class QtData(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        loadUi(os.path.join(path, 'resources', 'data.ui'), self)
        self.parent = parent

        self.btnConnect.clicked.connect(self.btnConnectClickced)
        self.btnJob.clicked.connect(self.btnJobClicked)
        self.btnPredict.clicked.connect(self.btnPredictClicked)

        home = os.path.expanduser('~')
        self.dirdata = os.path.join(home, 'bag_data')
        self.job = None

    def btnConnectClickced(self):
        self.save_access()

    def save_access(self):
        filename = os.path.join(self.dirdata, self.job, 'access.yalm')
        accessdata = {'IP': '172.20.0.204',
                      'user': str(self.txtUser.text()),
                      'password': str(self.txtPassword.text())}
        with open(filename, 'w') as outfile:
            outfile.write(yaml.dump(accessdata, default_flow_style=True))

    def btnJobClicked(self):
        self.job = self.txtJobName.text()
        dirname = os.path.join(self.dirdata, self.job)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if self.parent:
            self.parent.workdir = dirname

    def btnPredictClicked(self):
        param = self.getMaterialParameters()
        rospy.set_param('/material', param)

    def setMaterialParameters(self, params):
        self.txtBaseMaterial.setText(params['base'])
        self.txtPowderMaterial.setText(params['powder'])

    def getMaterialParameters(self):
        params = {'base': self.txtBaseMaterial.text(),
                  'powder': self.txtPowderMaterial.text()}
        return params


if __name__ == "__main__":
    rospy.init_node('data_panel')

    app = QtGui.QApplication(sys.argv)
    qt_data = QtData()
    qt_data.show()
    app.exec_()
