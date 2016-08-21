#!/usr/bin/env python
import os
import rviz
import rospy
import rospkg

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore

from mashes_measures.msg import MsgStatus

from qt_data import QtData
from qt_param import QtParam


path = rospkg.RosPack().get_path('cyplam_robviz')


class MyViz(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        ## rviz.VisualizationFrame is the main container widget of the
        ## regular RViz application. In this example, we disable everything
        ## so that the only thing visible is the 3D render window.
        self.frame = rviz.VisualizationFrame()
        self.frame.setSplashPath("")
        self.frame.initialize()

        # Read the configuration from the config file for visualization.
        reader = rviz.YamlConfigReader()
        config = rviz.Config()

        reader.readFile(config, os.path.join(path, 'config', 'workcell.rviz'))
        self.frame.load(config)

        self.setWindowTitle(config.mapGetChild("Title").getValue())

        self.frame.setMenuBar(None)
        self.frame.setHideButtonVisibility(False)

        self.manager = self.frame.getManager()
        self.grid_display = self.manager.getRootDisplayGroup().getDisplayAt(0)

        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(9, 0, 9, 0)
        self.setLayout(layout)

        h_layout = QtGui.QHBoxLayout()
        layout.addLayout(h_layout)

        orbit_button = QtGui.QPushButton("Orbit View")
        orbit_button.clicked.connect(self.onOrbitButtonClick)
        h_layout.addWidget(orbit_button)

        front_button = QtGui.QPushButton("Front View")
        front_button.clicked.connect(self.onFrontButtonClick)
        h_layout.addWidget(front_button)

        right_button = QtGui.QPushButton("Rigth View")
        right_button.clicked.connect(self.onRightButtonClick)
        h_layout.addWidget(right_button)

        top_button = QtGui.QPushButton("Top View")
        top_button.clicked.connect(self.onTopButtonClick)
        h_layout.addWidget(top_button)

        layout.addWidget(self.frame)

    ## switchToView() works by looping over the views saved in the
    ## ViewManager and looking for one with a matching name.
    def switchToView(self, view_name):
        view_man = self.manager.getViewManager()
        for i in range(view_man.getNumViews()):
            if view_man.getViewAt(i).getName() == view_name:
                view_man.setCurrentFrom(view_man.getViewAt(i))
                return
        print("Did not find view named %s." % view_name)

    def onOrbitButtonClick(self):
        self.switchToView("Orbit View")

    def onFrontButtonClick(self):
        self.switchToView("Front View")

    def onRightButtonClick(self):
        self.switchToView("Right View")

    def onTopButtonClick(self):
        self.switchToView("Top View")


class Robviz(QtGui.QMainWindow):
    def __init__(self):
        super(Robviz, self).__init__()
        loadUi(os.path.join(path, 'resources', 'robviz.ui'), self)

        rospy.Subscriber(
            '/supervisor/status', MsgStatus, self.cbStatus, queue_size=1)

        self.boxPlot.addWidget(MyViz())

        home = os.path.expanduser('~')
        self.workdir = os.path.join(home, 'data')

        self.qtData = QtData()
        self.qtParam = QtParam()

        self.tabWidget.addTab(self.qtData, 'Data')
        self.tabWidget.addTab(self.qtParam, 'Parameters')

        self.btnQuit.setIcon(QtGui.QIcon.fromTheme('application-exit'))
        self.btnQuit.clicked.connect(self.btnQuitClicked)

        self.speed = 0
        self.power = 0
        self.running = False
        self.laser_on = False

        tmrInfo = QtCore.QTimer(self)
        tmrInfo.timeout.connect(self.updateStatus)
        tmrInfo.start(100)

    def cbStatus(self, msg_status):
        self.running = msg_status.running
        self.laser_on = msg_status.laser_on
        self.speed = msg_status.speed
        self.power = msg_status.power

    def updateStatus(self):
        self.lblSpeed.setText("Speed: %.1f mm/s" % (self.speed))
        self.lblPower.setText("Power: %i W" % (self.power))
        if self.running:
            self.lblStatus.setText('Running')
            self.lblStatus.setStyleSheet(
                "background-color: rgb(0, 255, 0); color: rgb(0, 0, 0);")
        else:
            self.lblStatus.setText('Stopped')
            self.lblStatus.setStyleSheet(
                "background-color: rgb(255, 0, 0); color: rgb(0, 0, 0);")
        if self.laser_on:
            self.lblLaser.setText('Laser ON')
            self.lblLaser.setStyleSheet(
                "background-color: rgb(255, 255, 0); color: rgb(0, 0, 0);")
        else:
            self.lblLaser.setText('Laser OFF')
            self.lblLaser.setStyleSheet(
                "background-color: rgb(0, 0, 255); color: rgb(0, 0, 0);")

    def btnQuitClicked(self):
        QtCore.QCoreApplication.instance().quit()


if __name__ == '__main__':
    import sys

    rospy.init_node('robviz')

    app = QtGui.QApplication(sys.argv)
    robviz = Robviz()
    robviz.show()
    sys.exit(app.exec_())
