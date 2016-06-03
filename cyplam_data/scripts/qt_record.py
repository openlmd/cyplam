#!/usr/bin/env python
import os
import sys
import time
import signal

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore

from move_data.move_data import move_file


class QtRecord(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        # Layout are better for placing widgets
        layout = QtGui.QVBoxLayout()
        self.runButton = QtGui.QPushButton('Run')
        self.runButton.clicked.connect(self.callProgram)

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

    def dataReady(self):
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.End)
        text = str(self.process.readAll())
        #text = text.splitlines()
        cursor.insertText(text)
        self.output.ensureCursorVisible()

    def callProgram(self):
        #self.process.start('./test.sh')
        print os.getcwd()
        os.chdir('/home/jorge/bag_data')
        print os.getcwd()
        filename = 'data_' + time.strftime('%Y%m%d-%H%M%S') + '.bag'
        self.process.start('rosrun rosbag record -O %s /tachyon/geometry /control/power' % filename)

    def killProgram(self):
        os.system('killall -2 record')
        #print self.process.state()
        move_file('/home/jorge/bag_data/', '/home/ryco/data/')


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    qt_record = QtRecord()
    qt_record.show()
    app.exec_()
