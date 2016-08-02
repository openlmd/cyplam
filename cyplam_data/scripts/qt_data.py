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
DIRDEST = '/home/ryco/data/'

TOPICS = ['/tachyon/image',
          '/tachyon/geometry',
          #'/control/power',
          '/joint_states']

PARAMS = ['/material',
          '/powder',
          '/process']


class QtData(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        path = rospkg.RosPack().get_path('cyplam_data')
        loadUi(os.path.join(path, 'resources', 'data.ui'), self)

        rospy.Subscriber(
            '/supervisor/status', MsgStatus, self.cbStatus, queue_size=1)

        self.btnConnect.clicked.connect(self.btnConnectClicked)
        self.btnJob.clicked.connect(self.btnJobClicked)
        self.btnPredict.clicked.connect(self.btnPredictClicked)
        self.btnRecord.clicked.connect(self.btnRecordClicked)
        self.btnTransfer.clicked.connect(self.btnTransferClicked)

        dirdata = os.path.join(HOME, DIRDATA)
        if not os.path.exists(dirdata):
            os.mkdir(dirdata)

        self.job = ''
        self.name = ''
        self.status = False
        self.running = False
        self.process = QtCore.QProcess(self)

        if rospy.has_param('/material'):
            self.setMaterialParameters(rospy.get_param('/material'))

        self.btnJobClicked()

    def btnConnectClicked(self):
        self.txtOutput.textCursor().insertText('> trying to connect.\n')
        accessfile = os.path.join(HOME, DIRDATA, 'access.yalm')
        if test_connection(accessfile):
            self.txtOutput.textCursor().insertText('> connected.\n')
            self.btnTransfer.setEnabled(True)
        else:
            self.txtOutput.textCursor().insertText('> not connected.\n')
            self.btnTransfer.setEnabled(False)

    def btnJobClicked(self):
        self.job = self.txtJobName.text()
        dirname = os.path.join(HOME, DIRDATA, self.job)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            self.dirdata = dirname
        else:
            try:
                name, num = self.job[:-4], int(self.job[-4:]) + 1
            except:
                name, num = self.job, 1
            self.txtJobName.setText('%s%04i' % (name, num))
            self.btnJobClicked()

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
            self.btnRecord.setText('Recording...')
            self.txtOutput.textCursor().insertText('> ready for data.\n')
        else:
            self.btnRecord.setText('Record Data')
            self.txtOutput.textCursor().insertText('> stopped.\n')

    def saveParameters(self):
        filename = 'param_' + self.name + '.yaml'
        params = {param[1:]: rospy.get_param(param) for param in PARAMS}
        with open(os.path.join(self.dirdata, filename), 'w') as outfile:
            outfile.write(yaml.dump(params, default_flow_style=False))
        self.txtOutput.textCursor().insertText(str(params))

    def dataReady(self):
        cursor = self.txtOutput.textCursor()
        cursor.movePosition(cursor.End)
        text = str(self.process.readAll())
        cursor.insertText(text)
        self.txtOutput.moveCursor(QtGui.QTextCursor.End)
        self.txtOutput.ensureCursorVisible()

    def callProgram(self):
        os.chdir(self.dirdata)
        filename = 'data_' + self.name + '.bag'
        self.process.start(
            'rosrun rosbag record -O %s %s' % (filename, ' '.join(TOPICS)))

    def killProgram(self):
        os.system('killall -2 record')
        self.process.waitForFinished()

    def cbStatus(self, msg_status):
        if self.running:
            status = msg_status.running
            if not self.status and status:
                self.name = time.strftime('%Y%m%d-%H%M%S')
                self.saveParameters()
                self.txtOutput.textCursor().insertText(
                    '> recording topics:\n%s\n' % '\n'.join(TOPICS))
                self.callProgram()
            elif self.status and not status:
                self.killProgram()
                self.txtOutput.textCursor().insertText(
                    '> %s recorded.\n' % self.name)
            self.status = status

    def btnTransferClicked(self):
        accessfile = os.path.join(HOME, DIRDATA, 'access.yalm')
        filename = 'data_' + self.name + '.bag'
        if not self.name == '':
            self.btnTransfer.setEnabled(False)
            try:
                self.txtOutput.textCursor().insertText(
                    '> trying to transfer file %s.\n' % filename)
                move_file(accessfile,
                          os.path.join(self.dirdata, filename), DIRDEST)
                self.txtOutput.textCursor().insertText(
                    '> file %s transfered.\n' % filename)
            except:
                self.txtOutput.textCursor().insertText(
                    '> transference failed.\n')
            self.btnTransfer.setEnabled(True)
        else:
            self.txtOutput.textCursor().insertText('> transference failed.\n')


if __name__ == "__main__":
    rospy.init_node('data_panel')

    app = QtGui.QApplication(sys.argv)
    qt_data = QtData()
    qt_data.show()
    app.exec_()
