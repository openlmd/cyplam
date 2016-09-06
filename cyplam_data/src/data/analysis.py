import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from measures.velocity import Velocity
from measures.geometry import Geometry
from otsu import Otsu

def serialize_frame(frame, encode='*.png'):
    return cv2.imencode(encode, frame)[1].tostring(None)


def deserialize_frame(string):
    return cv2.imdecode(
        np.fromstring(string, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)


def read_frames(frames):
    return np.array([deserialize_frame(frame) for frame in frames])


def write_hdf5(filename, data, keys=None):
    store = pd.HDFStore(filename, complevel=9, complib='blosc')
    if keys is None:
        keys = data.keys()
    for key in keys:
        store.put(key, data[key], format='table', append=False)
    store.close()


def read_hdf5(filename, keys=None):
    store = pd.HDFStore(filename)
    if keys is None:
        keys = [key[1:] for key in store.keys()]
    data = {}
    for key in keys:
        data[key] = store.get(key)
    store.close()
    return data


def plot_image(img):
    plt.figure()
    plt.imshow(img, interpolation='none', cmap='jet', vmin=0, vmax=1024)
    plt.show()


def plot_histogram(img):
    plt.figure()
    plt.subplot(211)
    plt.imshow(img, interpolation='none', cmap='jet', vmin=0, vmax=1024)
    plt.subplot(212)
    plt.hist(img.flatten(), 100, range=(0, 1024), fc='k', ec='k')
    plt.xlim(0, 1024)
    plt.show()


def plot_image3d(img):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    ax.plot_surface(xx, yy, img, rstride=1, cstride=1, linewidth=0,
                    cmap='jet', vmin=0, vmax=1024)
    plt.show()


def calculate_velocity(time, position):
    velocity = Velocity()
    data = {'speed': [], 'velocity': [], 'running': []}
    for k in range(len(position)):
        speed, vel = velocity.instantaneous(time[k], np.array(position[k]))
        data['speed'].append(speed)
        data['velocity'].append(vel)
        data['running'].append(speed > 0.0005)
    return data


def append_data(dataframe, data):
    for key, value in data.iteritems():
        dataframe[key] = pd.Series(value, index=dataframe.index)
    return dataframe


def find_geometry(tachyon):
    geometry = Geometry(200)
    ellipses = []
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        ellipse = geometry.find_geometry(img)
        ellipses.append(np.array(ellipse))
    ellipses = np.array(ellipses)
    widths = np.array([axis[1]for axis in ellipses[:, 1]])
    plt.figure()
    plt.subplot(211)
    plt.plot(widths)
    plt.subplot(212)
    plt.plot(widths * 2.66666667)
    plt.show()

    # plot_image(geometry.draw_geometry(img, ellipse))
    # plot_image3d(img)  # Show image as 3D surface
    # plot_histogram(img)

def draw_temperature(tachyon):
    geometry = Geometry(200)
    ellipses = []
    temp = []
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        ellipse = geometry.find_geometry(img)
        ellipses.append(np.array(ellipse))
    ellipses = np.array(ellipses)
    widths = np.array([axis[1]for axis in ellipses[:, 1]])
    if 'temperature' in tachyon.columns:
        for frame in tachyon.temperature:
            temp.append(frame)
    plt.figure()
    plt.subplot(211)
    plt.plot(widths)
    plt.subplot(212)
    plt.plot(temp)
    plt.show()

def find_track(tachyon):
    tachyonw = tachyon[tachyon.minor_axis.notnull()]
    laser = np.array(tachyonw['minor_axis'] > 0)
    lasernr = np.append(np.bitwise_not(laser[0]), np.bitwise_not(laser[:-1]))
    lasernl = np.append(np.bitwise_not(laser[1:]), np.bitwise_not(laser[-1]))
    laser_on = np.bitwise_and(laser, lasernr)
    laser_off = np.bitwise_and(laser, lasernl)
    laser_on_idx = tachyonw.iandex[laser_on]
    laser_off_idx = tachyonw.index[laser_off]
    print laser_on_idx, laser_off_idx
    plt.figure()
    plt.plot(laser_on)
    plt.plot(laser_off)
    plt.show()


def max_evolution(tachyon):
    maxims = []
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        maxims.append(img.max())
    maxims = np.array(maxims)
    print maxims.max()
    plt.figure()
    plt.plot(maxims)
    plt.show()


def center_evolution(tachyon):
    back = []
    temp = []
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        back.append(np.mean(img[15:18, 11:13]))
    back = np.array(back)
    print back.max()
    if 'temperature' in tachyon.columns:
        for i in tachyon.temperature:
            temp.append(i)
    plt.figure()
    plt.subplot(211)
    plt.plot(back)
    plt.subplot(212)
    # plt.ylim(30.0, 40.0)
    plt.plot(temp)
    plt.show()


def back_evolution(tachyon):
    back = []
    temp = []
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        back.append(np.mean(img[25:27, 25:27]))
    back = np.array(back)
    print back.max()
    if 'temperature' in tachyon.columns:
        for i in tachyon.temperature:
            temp.append(i)
    plt.figure()
    plt.subplot(211)
    plt.plot(back)
    plt.subplot(212)
    plt.ylim(30.0, 40.0)
    plt.plot(temp)
    plt.show()


def resize(scale, img, ellipse):
    img = cv2.resize(img, (scale*32, scale*32))
    ((x, y), (w, l), a) = ellipse
    ellipse = ((scale*x, scale*y), (scale*w, scale*l), a)
    return img, ellipse


def plot_frame(tachyon, nframe, th):
    img = deserialize_frame(tachyon.frame[nframe])
    geometry = Geometry(th)
    ellipse = geometry.find_geometry(img)
    plot_image(geometry.draw_geometry(img.copy(), ellipse))
    plot_image3d(img)  # Show image as 3D surface
    plot_histogram(img)


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
    data = read_hdf5(filename, tables)

    if 'robot' in data.keys():
        robot = data['robot']
        velocity = calculate_velocity(robot.time, robot.position)
        data['robot'] = append_data(robot, velocity)

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
            find_track(tachyon)

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
            find_geometry(tachyon)
