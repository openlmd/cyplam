#!/usr/bin/env python
import os
import sys
import time
import yaml
import rospy
import rospkg
import paramiko

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore

from mashes_measures.msg import MsgStatus
from move_data.move_data import move_file


TOPICS = ['/tachyon/image',
          '/tachyon/geometry',
          '/control/power',
          '/joint_states']


class QtRecord(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        path = rospkg.RosPack().get_path('cyplam_data')
        loadUi(os.path.join(path, 'resources', 'record.ui'), self)

        self.parent = parent

        self.btnRecord.clicked.connect(self.btnClickRecord)
        self.btnConnect.clicked.connect(self.btnClickConnect)
        self.btnTransfer.clicked.connect(self.btnClickTransfer)

        home = os.path.expanduser('~')
        self.dirdata = os.path.join(home, 'bag_data')
        if not os.path.exists(self.dirdata):
            os.mkdir(self.dirdata)

        self.filename = ''
        rospy.Subscriber(
            '/supervisor/status', MsgStatus, self.cbStatus, queue_size=1)
        self.status = False
        self.running = False
        self.process = QtCore.QProcess(self)
        #self.process.readyRead.connect(self.dataReady)

    def btnClickTransfer(self):
        if not self.filename == '':
            self.lbTransfer.setText('Trying to transfer %s' % self.filename)
            self.btnTransfer.setEnabled(False)
            try:
                move_file(os.path.join(self.dirdata, self.filename), '/home/ryco/data/')
            except:
                self.lbTransfer.setText('Transference failed')
            self.btnTransfer.setEnabled(True)
            self.txtOutput.textCursor().insertText('> %s transfered.\n' % self.filename)
            self.lbTransfer.setText('File %s transfered' % self.filename)
        else:
            self.lbTransfer.setText('Any file to transfer')

    def btnClickConnect(self):
        self.lbState.setText('Trying connection')
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            if self.parent:
                print 'parent', self.parent.workdir
                directory = os.path.join(self.parent.workdir, 'access.yalm')
            else:
                directory = '/home/panadeiro/bag_data/access.yalm'
            print 'directory', directory
            f = open(directory)
            accessdata = yaml.load(f)
            ssh.connect(accessdata['IP'], username=accessdata['user'], password=accessdata['password'])
            self.lbState.setText('Connected')
            self.btnTransfer.setEnabled(True)
        except paramiko.SSHException:
            print "Connection Failed"
            self.lbState.setText('Connection failed, check your access data')
            self.btnTransfer.setEnabled(False)
            # quit()

        ssh.close()

    def btnClickRecord(self):
        if self.parent:
            self.dirdata = self.parent.workdir
            print 'ahora dirdata', self.dirdata
        self.running = not self.running
        if self.running:
            self.btnRecord.setText('Ready for Data')
            self.txtOutput.textCursor().insertText(
                str(rospy.get_param('/powder'))+'\n')
            self.txtOutput.textCursor().insertText(
                str(rospy.get_param('/process'))+'\n')
            self.txtOutput.textCursor().insertText('> ready for data.\n')
        else:
            self.btnRecord.setText('Record Data')
            self.txtOutput.textCursor().insertText('> stopped.\n')


    def cbStatus(self, msg_status):
        if not self.status and msg_status.running:
            if self.running:
                self.btnRecord.setText('Recording...')
                self.txtOutput.textCursor().insertText('> recording topics:\n%s\n' % '\n'.join(TOPICS))
                self.callProgram()
                self.saveParameters()
        elif self.status and not msg_status.running:
            self.running = False
            self.killProgram()
            self.btnRecord.setText('Record Data')
            self.txtOutput.textCursor().insertText('> %s recorded.\n' % self.filename)
        self.status = msg_status.running

    def saveParameters(self):
        fileparam = 'param_' + time.strftime('%Y%m%d-%H%M%S') + '.yaml'
        data = dict(
            material=dict(rospy.get_param('/material')),
            powder=dict(rospy.get_param('/powder')),
            process=dict(rospy.get_param('/process')),
            # control=dict(rospy.get_param('/control')),
        )
        print data
        with open(os.path.join(self.dirdata, fileparam), 'w') as outfile:
            outfile.write(yaml.dump(data, default_flow_style=True))

    def dataReady(self):
        cursor = self.txtOutput.textCursor()
        cursor.movePosition(cursor.End)
        text = str(self.process.readAll())
        cursor.insertText(text)
        self.txtOutput.ensureCursorVisible()

    def callProgram(self):
        os.chdir(self.dirdata)
        print 'cando grava o directorio ', self.dirdata
        self.filename = 'data_' + time.strftime('%Y%m%d-%H%M%S') + '.bag'
        self.process.start(
            'rosrun rosbag record -O %s %s' % (self.filename, ' '.join(TOPICS)))
        print self.filename

    def killProgram(self):
        os.system('killall -2 record')
        self.process.waitForFinished()
        self.lbTransfer.setText('File created')

if __name__ == '__main__':
    rospy.init_node('record_panel')
    app = QtGui.QApplication(sys.argv)
    qt_record = QtRecord()
    qt_record.show()
    app.exec_()
