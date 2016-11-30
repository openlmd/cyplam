#!/usr/bin/env python
import os
import sys
import rospy
import rospkg

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore


class QtParam(QtGui.QWidget):
    accepted = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        path = rospkg.RosPack().get_path('cyplam_param')
        loadUi(os.path.join(path, 'resources', 'param.ui'), self)

        self.btnAccept.clicked.connect(self.btnAcceptClicked)

        self.setPowderParameters(rospy.get_param('/powder'))
        self.setProcessParameters(rospy.get_param('/process'))

    def getPowderParameters(self):
        params = {'shield': self.sbShield.value(),
                  'carrier': self.sbCarrier.value(),
                  'stirrer': self.sbStirrer.value(),
                  'turntable': self.sbTurntable.value()}
        return params

    def setPowderParameters(self, params):
        self.sbShield.setValue(params['shield'])
        self.sbCarrier.setValue(params['carrier'])
        self.sbStirrer.setValue(params['stirrer'])
        self.sbTurntable.setValue(params['turntable'])

    def getProcessParameters(self):
        params = {'speed': self.sbSpeed.value(),
                  'power': self.sbPower.value(),
                  'focus': self.sbFocus.value()}
        return params

    def setProcessParameters(self, params):
        self.sbSpeed.setValue(params['speed'])
        self.sbPower.setValue(params['power'])
        self.sbFocus.setValue(params['focus'])

    def btnAcceptClicked(self):
        powder = self.getPowderParameters()
        rospy.set_param('/powder', powder)
        process = self.getProcessParameters()
        rospy.set_param('/process', process)
        self.accepted.emit([])


if __name__ == "__main__":
    rospy.init_node('parameters_panel')

    app = QtGui.QApplication(sys.argv)
    qt_param = QtParam()
    qt_param.show()
    app.exec_()
