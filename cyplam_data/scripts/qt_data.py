#!/usr/bin/env python
import os
import sys
import time
import yaml
import rospy
import rospkg

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore

from mashes_measures.msg import MsgStatus

from data.move_data import move_file
from data.move_data import test_connection

HOME = os.path.expanduser('~')
DIRDATA = 'bag_data'
DIRDEST = 'data'

TOPICS = ['/tachyon/image',
          '/camera/image',
          '/tachyon/temperature',
          #'/tachyon/geometry',
          #'/control/power',
          '/ueye/cloud',
          '/joint_states']

PARAMS = ['/material',
          '/powder',
          '/process',
          '/routine']


class QtData(QtGui.QWidget):
    accepted = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        path = rospkg.RosPack().get_path('cyplam_data')
        loadUi(os.path.join(path, 'resources', 'data.ui'), self)

        rospy.Subscriber(
            '/supervisor/status', MsgStatus, self.cbStatus, queue_size=1)

        self.btnSynchronize.clicked.connect(self.btnSynchronizeClicked)
        self.btnJob.clicked.connect(self.btnJobClicked)
        self.btnPredict.clicked.connect(self.btnPredictClicked)
        self.btnRecord.clicked.connect(self.btnRecordClicked)

        dirdata = os.path.join(HOME, DIRDATA)
        if not os.path.exists(dirdata):
            os.mkdir(dirdata)

        self.job = ''
        self.name = ''
        self.status = False
        self.running = False
        self.recording = False
        self.process = QtCore.QProcess(self)

        if rospy.has_param('/material'):
            self.setMaterialParameters(rospy.get_param('/material'))

        self.btnJobClicked()

    def btnSynchronizeClicked(self):
        os.system("gnome-terminal --working-directory=~ -e 'rsync -av --progress bag_data/ ryco@172.20.0.204:data'")

    def btnJobClicked(self):
        self.job = self.txtJobName.text()
        dirname = os.path.join(HOME, DIRDATA, self.job)
        self.dirdata = dirname

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

    def btnRecordClicked(self):
        self.running = not self.running
        if self.running:
            self.btnJobClicked()
            self.btnRecord.setText('Recording...')
            if not os.path.exists(self.dirdata):
                os.mkdir(self.dirdata)
            self.txtOutput.textCursor().insertText('> ready for data.\n')
            self.accepted.emit([])
        else:
            self.btnRecord.setText('Record Data')
            self.txtOutput.textCursor().insertText('> stopped.\n')

    def saveParameters(self):
        filename = self.name + '.yaml'
        params = {param[1:]: rospy.get_param(param) for param in PARAMS}
        with open(os.path.join(self.dirdata, filename), 'w') as outfile:
            outfile.write(yaml.dump(params, default_flow_style=False))
        self.txtOutput.textCursor().insertText(str(params))

    def saveRoutine(self):
        filename = self.name + '.jas'
        routine = rospy.get_param('/routine')
        with open(os.path.join(self.dirdata, filename), 'w') as outfile:
            outfile.write(routine)

    def callProgram(self):
        os.chdir(self.dirdata)
        filename = self.name + '.bag'
        self.recording = True
        self.process.start(
            'rosrun rosbag record -O %s %s' % (filename, ' '.join(TOPICS)))

    def killProgram(self):
        os.system('killall -2 record')
        self.process.waitForFinished()
        self.recording = False

    def cbStatus(self, msg_status):
        if self.running:
            status = msg_status.running
            if not self.status and status and not self.recording:
                self.name = time.strftime('%Y%m%d-%H%M%S')
                self.saveParameters()
                self.txtOutput.textCursor().insertText(
                    '> recording topics:\n%s\n' % '\n'.join(TOPICS))
                self.saveRoutine()
                self.callProgram()
            elif self.status and not status:
                self.killProgram()
                self.txtOutput.textCursor().insertText(
                    '> %s recorded.\n' % self.name)
            self.status = status
            self.txtOutput.moveCursor(QtGui.QTextCursor.End)
            self.txtOutput.ensureCursorVisible()


if __name__ == "__main__":
    rospy.init_node('data_panel')

    app = QtGui.QApplication(sys.argv)
    qt_data = QtData()
    qt_data.show()
    app.exec_()
