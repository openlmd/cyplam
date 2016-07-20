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


TOPICS = ['/tachyon/image',
          '/tachyon/geometry',
          '/control/power',
          '/joint_states']

PARAMS = ['/material',
          '/powder',
          '/process']


class QtData(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        path = rospkg.RosPack().get_path('cyplam_data')
        loadUi(os.path.join(path, 'resources', 'data.ui'), self)

        self.btnConnect.clicked.connect(self.btnConnectClicked)
        self.btnJob.clicked.connect(self.btnJobClicked)
        self.btnPredict.clicked.connect(self.btnPredictClicked)
        self.btnRecord.clicked.connect(self.btnRecordClicked)
        self.btnTransfer.clicked.connect(self.btnTransferClicked)

        home = os.path.expanduser('~')
        self.dirdata = os.path.join(home, 'bag_data')
        if not os.path.exists(self.dirdata):
            os.mkdir(self.dirdata)

        self.name = ''
        self.job = None
        self.status = False
        self.running = False
        self.process = QtCore.QProcess(self)

        rospy.Subscriber(
            '/supervisor/status', MsgStatus, self.cbStatus, queue_size=1)

        if rospy.has_param('/material'):
            self.setMaterialParameters(rospy.get_param('/material'))

    def btnConnectClicked(self):
        self.lblInfo.setText('Trying to connect')
        accessfile = os.path.join(self.dirdata, 'access.yalm')
        if test_connection(accessfile):
            self.lblInfo.setText('Connected')
            self.btnTransfer.setEnabled(True)
        else:
            self.lblInfo.setText('Not connected')
            self.btnTransfer.setEnabled(False)

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

    def btnRecordClicked(self):
        self.running = not self.running
        if self.running:
            self.btnRecord.setText('Ready for Data')
            [self.txtOutput.textCursor().insertText(
                str(rospy.get_param(param))+'\n') for param in PARAMS]
            self.txtOutput.textCursor().insertText('> ready for data.\n')
        else:
            self.btnRecord.setText('Record Data')
            self.txtOutput.textCursor().insertText('> stopped.\n')

    def saveParameters(self):
        filename = 'param_' + self.name + '.yaml'
        params = {param[1:]: rospy.get_param(param) for param in PARAMS}
        with open(os.path.join(self.dirdata, filename), 'w') as outfile:
            outfile.write(yaml.dump(params, default_flow_style=False))

    def dataReady(self):
        cursor = self.txtOutput.textCursor()
        cursor.movePosition(cursor.End)
        text = str(self.process.readAll())
        cursor.insertText(text)
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
        status = msg_status.running
        if not self.status and status:
            if self.running:
                self.name = time.strftime('%Y%m%d-%H%M%S')
                self.saveParameters()
                self.btnRecord.setText('Recording...')
                self.txtOutput.textCursor().insertText(
                    '> recording topics:\n%s\n' % '\n'.join(TOPICS))
                self.callProgram()
        elif self.status and not status:
            self.killProgram()
            self.btnRecord.setText('Record Data')
            self.txtOutput.textCursor().insertText(
                '> %s recorded.\n' % self.name)
        self.status = status

    def btnTransferClicked(self):
        accessfile = os.path.join(self.dirdata, 'access.yalm')
        filename = 'data_' + self.name + '.bag'
        if not self.name == '':
            self.lblInfo.setText('Trying to transfer %s' % filename)
            self.btnTransfer.setEnabled(False)
            try:
                move_file(accessfile, os.path.join(self.dirdata, filename), '/home/ryco/data/')
            except:
                self.lblInfo.setText('Transference failed')
            self.btnTransfer.setEnabled(True)
            self.txtOutput.textCursor().insertText('> %s transfered.\n' % filename)
            self.lblInfo.setText('File %s transfered' % filename)
        else:
            self.lblInfo.setText('No file to transfer')


if __name__ == "__main__":
    rospy.init_node('data_panel')

    app = QtGui.QApplication(sys.argv)
    qt_data = QtData()
    qt_data.show()
    app.exec_()
