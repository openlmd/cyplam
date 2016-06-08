#!/usr/bin/env python
import os
import sys
import time
import rospy

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore

from move_data.move_data import move_file
from mashes_measures.msg import MsgStatus



class QtRecord(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.first = True
        # Layout are better for placing widgets
        layout = QtGui.QVBoxLayout()
        self.runButton = QtGui.QPushButton('Run')
        self.runButton.clicked.connect(self.callProgram)

        rospy.Subscriber('/supervisor/status', MsgStatus, self.status)

        self.killButton = QtGui.QPushButton('Kill')
        self.killButton.clicked.connect(self.killProgram)

        self.output = QtGui.QTextEdit()

        layout.addWidget(self.output)
        layout.addWidget(self.runButton)
        layout.addWidget(self.killButton)

        self.setLayout(layout)

        # QProcess object for external app
        self.process = QtCore.QProcess(self)
        # QProcess emits `readyRead` when there is data to be read
        self.process.readyRead.connect(self.dataReady)

        # Just to prevent accidentally running multiple times
        # Disable the button when process starts, and enable it when it finishes
        self.process.started.connect(lambda: self.runButton.setEnabled(False))
        self.process.finished.connect(lambda: self.runButton.setEnabled(True))

    def status(self, msg_status):
        if self.first:
            self.first = False
            self.status_ant = msg_status.running
        else:
            if (self.status_ant != msg_status.running):
                if msg_status.running:
                    self.callProgram()
                else:
                    self.killProgram()
            self.status_ant = msg_status.running

    def dataReady(self):
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.End)
        text = str(self.process.readAll())
        cursor.insertText(text)
        self.output.ensureCursorVisible()

    def callProgram(self):
        print os.getcwd()
        os.chdir('/home/panadeiro/bag_data/')
        print os.getcwd()
        filename = 'data_' + time.strftime('%Y%m%d-%H%M%S') + '.bag'
        self.process.start('rosrun rosbag record -O %s /tachyon/geometry /control/power' % filename)

    def killProgram(self):
        os.system('killall -2 record')
        self.process.waitForFinished()
        move_file('/home/panadeiro/bag_data/', '/home/ryco/data/')
        self.runButton.setEnabled(True)


if __name__ == '__main__':
    rospy.init_node('record_panel')
    app = QtGui.QApplication(sys.argv)
    qt_record = QtRecord()
    qt_record.show()
    app.exec_()
