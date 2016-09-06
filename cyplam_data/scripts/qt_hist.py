#!/usr/bin/env python
import sys
import rospy
import numpy as np

from python_qt_binding import QtGui
from python_qt_binding import QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import gridspec


from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class QtHist(QtGui.QWidget):
    def __init__(self, parent=None):
        super(QtHist, self).__init__(parent)
        self.fig = Figure(figsize=(9, 3), dpi=72, facecolor=(0.76, 0.78, 0.8),
                          edgecolor=(0.1, 0.1, 0.1), linewidth=2)
        self.canvas = FigureCanvas(self.fig)
        gs = gridspec.GridSpec(5, 1)
        self.plot1_axis1 = self.fig.add_subplot(gs[0:4, 0])
        self.plot1_axis1.get_yaxis().set_ticklabels([])

        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.canvas)

        self.bridge = CvBridge()
        rospy.Subscriber('/tachyon/image', Image, self.image_hist)

        self.first = False

    def image_hist(self, msg_image):
        frame = self.bridge.imgmsg_to_cv2(msg_image)
        self.image = frame
        self.first = True

    def timeMeasuresEvent(self):
            if self.first:
                self.plot1_axis1.hist(self.image.flatten(), 100,
                                      range=(0, 1024),
                                      fc='k', ec='k')
            self.canvas.draw()


if __name__ == "__main__":
    rospy.init_node('data_plot')
    app = QtGui.QApplication(sys.argv)
    qt_hist = QtHist()
    tmrMeasures = QtCore.QTimer()
    tmrMeasures.timeout.connect(qt_hist.timeMeasuresEvent)
    tmrMeasures.start(1000)
    qt_hist.show()
    app.exec_()
