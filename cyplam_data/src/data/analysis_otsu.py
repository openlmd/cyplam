import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from measures.velocity import Velocity
from measures.geometry import Geometry

from skimage import data
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

import analysis


def draw_otsu(tachyon):
    geometry = Geometry(200)
    ellipses = []
    otsu = []
    for frame in tachyon.frame:
        img = analysis.deserialize_frame(frame)
        ellipse = geometry.find_geometry(img)
        ellipses.append(np.array(ellipse))
        try:
            val = filters.threshold_otsu(img)
            otsu.append(val)
        except:
            otsu.append(126)
    ellipses = np.array(ellipses)
    widths = np.array([axis[1]for axis in ellipses[:, 1]])
    plt.figure()
    plt.subplot(211)
    plt.plot(widths)
    plt.subplot(212)
    plt.plot(otsu)
    plt.show()


def find_otsu_geometry(tachyon):
    ellipses = []
    ellipses_2 = []
    otsu = []
    for frame in tachyon.frame:
        img = analysis.deserialize_frame(frame)
        geometry = Geometry(200)
        ellipse_2 = geometry.find_geometry(img)
        try:
            val = filters.threshold_otsu(img)
        except:
            val = 127
        otsu.append(val)
        geometry = Geometry(val)
        ellipse = geometry.find_geometry(img)
        ellipses.append(np.array(ellipse))
        ellipses_2.append(np.array(ellipse_2))
    ellipses = np.array(ellipses)
    ellipses_2 = np.array(ellipses_2)
    widths = np.array([axis[1]for axis in ellipses[:, 1]])
    widths_2 = np.array([axis[1]for axis in ellipses_2[:, 1]])
    plt.figure()
    plt.subplot(311)
    plt.plot(widths_2)
    plt.subplot(312)
    plt.plot(otsu)
    plt.subplot(313)
    plt.plot(widths)
    plt.show()




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', type=str, default=None, help='hdf5 filename')
    parser.add_argument(
        '-t', '--tables', type=str, default=None, nargs='+', help='tables names')
    args = parser.parse_args()

    filename = args.file
    tables = args.tables
    data = analysis.read_hdf5(filename, tables)

    if 'robot' in data.keys():
        robot = data['robot']
        velocity = analysis.calculate_velocity(robot.time, robot.position)
        data['robot'] = analysis.append_data(robot, velocity)

        plt.figure()
        plt.subplot(211)
        plt.plot(velocity['speed'])
        plt.subplot(212)
        plt.plot(velocity['running'])
        plt.show()

    if 'tachyon' in data.keys():
        tachyon = data['tachyon']
        tachyon = tachyon[tachyon.frame.notnull()]
        #frames = read_frames(tachyon.frame)

        if 'minor_axis' in tachyon.columns:
            idx = tachyon.index[tachyon['minor_axis'] > 0]
            idx0, idx1 = idx[0], idx[-1]
            time0 = tachyon.time[idx0]
            time1 = tachyon.time[idx1]
            print idx0, idx1, time0, time1
            idx = tachyon.index[tachyon.time < time0-1]
            idx0 = idx[-1]
            idx = tachyon.index[tachyon.time > time1+1]
            idx1 = idx[0]
            print idx0, idx1
            if idx0 < 0:
                idx0 = 0
            if idx1 > len(tachyon):
                idx1 = len(tachyon)
            meas = tachyon.loc[idx0:idx1]
            time = np.array(meas['time'])
            analysis.find_track(tachyon)

            # N = len(measures['power'])
            # print measures.loc[N-10:N-1]['time']
            # time = np.array(measures['time'])
            # print np.isclose(time[1:], time[:-1], atol=1e-03, rtol=0)

            tachyonp = tachyon[tachyon.power.notnull()]
            #tachyon.time = tachyon.time - tachyon.time[0]

            plt.figure()
            plt.subplot(211)
            tachyonp.plot(x='time', y='minor_axis',
                          xlim=(time[0], time[-1]), ylim=(0, 5), color='blue')
            plt.subplot(212)
            tachyonp.plot(x='time', y='power',
                          xlim=(time[0], time[-1]), ylim=(0, 1500), color='red')
            plt.show()
        else:
            analysis.find_geometry(tachyon)
