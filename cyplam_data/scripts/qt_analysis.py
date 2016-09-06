#!/usr/bin/env python
import os
import sys
import rospy
import rospkg

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore


from qt_hist import QtHist
from qt_display import QtDisplay
from mashes_tachyon.msg import MsgTemperature

from calibrate.calibrate import Calibrate


class QtAnalysis(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        path = rospkg.RosPack().get_path('cyplam_data')
        loadUi(os.path.join(path, 'resources', 'analysis.ui'), self)

        self.hist = QtHist()
        self.vLayout.addWidget(self.hist)
        self.qt_display = QtDisplay()
        self.display.addWidget(self.qt_display)
        self.btnCalibrate.clicked.connect(self.btnCalibrateClicked)

        rospy.Subscriber('/tachyon/temperature', MsgTemperature,
                         self.upTemperature, queue_size=1)

        self.calibrate = Calibrate()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.timeoutRunning)
        self.timer.start(100)

    def timeoutRunning(self):
        self.qt_display.timeoutRunning()
        self.hist.timeMeasuresEvent()

    def upTemperature(self, msg_temp):
        self.lTemperature.setText("%.2f" % msg_temp.temperature)

    def btnCalibrateClicked(self):
        self.calibrate.run()
        self.lCalibration.setText('Calibration temperature:' +
                                  self.lTemperature.text())

if __name__ == "__main__":
    rospy.init_node('analysis_panel')

    app = QtGui.QApplication(sys.argv)
    qt_analysis = QtAnalysis()

    qt_analysis.show()
    app.exec_()
