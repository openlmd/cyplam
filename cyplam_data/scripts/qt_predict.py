#!/usr/bin/env python
import sys
import rospy
import numpy as np

from python_qt_binding import QtGui
from python_qt_binding import QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import gridspec
from matplotlib.lines import Line2D

from cyplam_data.msg import MsgPredictv
from cyplam_data.msg import MsgPredictp

# class Filter():
#     def __init__(self, fc=100):
#         self.fc = fc
#         self.y = 0
#         self.t = 0
#
#     def update(self, x, t):
#         DT = t - self.t
#         a = (2 * np.pi * DT * self.fc) / (2 * np.pi * DT * self.fc + 1)
#         y = a * x + (1 - a) * self.y
#         self.y = y
#         self.t = t
#         return y


class QtPlot(QtGui.QWidget):
    def __init__(self, parent=None):
        super(QtPlot, self).__init__(parent)
        self.fig = Figure(figsize=(9, 6), dpi=72, facecolor=(0.76, 0.78, 0.8),
                          edgecolor=(0.1, 0.1, 0.1), linewidth=2)
        self.canvas = FigureCanvas(self.fig)
        pos1 = [0, 2]
        pos2 = [3, 5]
        gs = gridspec.GridSpec(5, 1)
        self.velocity = Graph(gs, pos1, self.fig, 'b', self.canvas)
        self.velocity.plot.get_yaxis().set_ticklabels([])
        self.power = Graph(gs, pos2, self.fig, 'r', self.canvas)

        self.plot_def()

        rospy.Subscriber('/predict/power', MsgPredictv, self.power_fn)
        rospy.Subscriber('/predict/velocity', MsgPredictp, self.velocity_fn)

        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.canvas)

    def plot_def(self):
        self.velocity.min = 0
        self.velocity.max = 20
        self.velocity.plot_title = 'Velocity prediction'
        self.velocity.yunits_title = 'Velocity [mm/s]'
        self.velocity.xunits_title = 'Time[s]'

        self.velocity.draw_figure()

        self.velocity.max_mea = 15.0
        self.velocity.min_mea = 0.5
        self.velocity.t = 1
        self.velocity.color = 'b'

        self.power.max_mea = 1900
        self.power.min_mea = 100
        self.power.color = 'r'

        self.power.min = 0
        self.power.max = 2000
        self.power.plot_title = 'Power prediction'
        self.power.yunits_title = 'Power[W]'
        self.power.xunits_title = 'Time[s]'
        self.power.draw_figure()

    def velocity_fn(self, msg_velocity):
        time = msg_velocity.header.stamp.to_sec()
        data = msg_velocity.value
        self.velocity.update_data(time, data)

    def power_fn(self, msg_power):
        time = msg_power.header.stamp.to_sec()
        data = msg_power.value
        self.power.update_data(time, data)

    def timeMeasuresEvent(self):
        self.velocity.timeEvent()
        self.power.timeEvent()

    def resizeEvent(self, event):
        self.velocity.figbackground = None
        self.velocity.background = None
        self.power.background = None


class Graph():
    def __init__(self, gs, pos, fig, col, canvas, parent=None):
        self.first = 0
        self.time_ant = 0
        self.distance = 0
        self.duration = 6
        self.buff_max = self.duration * 500
        self.min = 0
        self.max = 1000
        self.max_mea = 100
        self.min_mea = 1900
        self.color = col
        self.t = 1
        self.canvas = canvas
        self.fig = fig
        self.plot = self.fig.add_subplot(gs[pos[0]:pos[1], 0])

        self.reset_data()

        self.line = Line2D(
            self.time, self.data, color=self.color, linewidth=2, animated=True)
        self.text = self.plot.text(
            self.duration-10, 0, '', size=13, ha='right', va='center',
            backgroundcolor='w', color=self.color, animated=True)

        self.plot_title = 'Graph title'
        self.yunits_title = 'Y Units ()'
        self.xunits_title = 'X Units()'
        self.draw_figure()

    def reset_data(self):
        self.data = []
        self.time = []
        # self.filter = Filter()

    def draw_figure(self):
        self.plot.cla()
        self.plot.set_title(self.plot_title)

        self.plot.add_line(self.line)
        self.plot.set_xlim(0, self.duration)
        self.plot.set_ylabel(self.yunits_title)
        self.plot.set_xlabel(self.xunits_title)
        self.plot.set_ylim(self.min, self.max)
        self.plot.grid(True)

        self.canvas.draw()

        self.figbackground = self.canvas.copy_from_bbox(self.fig.bbox)
        self.background = self.canvas.copy_from_bbox(self.plot.bbox)

    def _limited_range(self, value, min_value, max_value):
        if value < min_value:
            value = min_value
        elif value > max_value:
            value = max_value
        return value

    def resize(self):
        self.figbackground = None
        self.background = None

    def timeEvent(self):
        if self.figbackground is None or self.background is None:
            self.draw_figure()
        try:
            self.canvas.restore_region(self.figbackground)
            self.canvas.restore_region(self.background)
            self.plot.draw_artist(self.line)
            self.plot.draw_artist(self.text)
            self.canvas.blit(self.plot.bbox)
        except:
            pass

    def buffers(self):
        self.time = self.time[-(self.buff_max-1):]
        self.data = self.data[-(self.buff_max-1):]

    def update_data(self, time, data):
        if self.first == 0 or self.first > time or self.time_ant > time:
            self.first = time
            self.reset_data()
            self.distance = 0
            self.canvas.update()
        if time-self.first > self.duration:
            self.distance = time-self.first-self.duration
        self.time.append(time-self.first)
        self.data.append(data)
        self.line.set_data(np.array(self.time)-self.distance, self.data)
        if len(self.data) > self.buff_max:
            self.buffers()
        if len(self.time) > 2:
            # Measures filtered value
            # mean = np.round(self.filter.update
            #                 (self.data[-1], self.time[-1]), self.t)
            mean = self.data[-1]
            self.text.set_text('%.1f' % mean)
            mean = self._limited_range(mean, self.min_mea, self.max_mea)
            self.text.set_y(mean)
            self.text.set_x(0.98 * self.duration)
        self.time_ant = time

if __name__ == "__main__":
    rospy.init_node('prediction_plot')
    app = QtGui.QApplication(sys.argv)
    qt_plot = QtPlot()
    tmrMeasures = QtCore.QTimer()
    tmrMeasures.timeout.connect(qt_plot.timeMeasuresEvent)
    tmrMeasures.start(100)
    qt_plot.show()
    app.exec_()
