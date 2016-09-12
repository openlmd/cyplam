import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from measures.velocity import Velocity
from measures.geometry import Geometry


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


def append_data(dataframe, data):
    for key, value in data.iteritems():
        dataframe[key] = pd.Series(value, index=dataframe.index)
    return dataframe


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


def plot_frame(frame, thr):
    img = deserialize_frame(frame)
    geometry = Geometry(thr)
    ellipse = geometry.find_geometry(img)
    plot_image(geometry.draw_geometry(img.copy(), ellipse))
    plot_image3d(img)  # Show image as 3D surface
    plot_histogram(img)


def plot_data(robot, x, y, color):
    rows = len(y)
    plt.figure()
    for k in range(rows):
        plt.subplot(rows, 1, k+1)
        plt.plot(robot[x[k]], robot[y[k]], color=color[k])
        plt.xlim(robot[x[k]].min(), robot[x[k]].max())
        plt.ylabel(y[k])
    plt.show()


def plot_speed(robot):
    plot_data(robot, x=('time', 'time'), y=('speed', 'running'),
              color=('blue', 'red'))


def plot_geometry(data):
    plot_data(data, x=('time', 'time'), y=('width', 'height'),
              color=('blue', 'green'))


def plot_power(data):
    plt.figure()
    plt.subplot(211)
    data.plot(x='time', y='minor_axis',
              xlim=(data['time'][0], data['time'][-1]), ylim=(0, 5),
              color='blue')
    plt.subplot(212)
    data.plot(x='time', y='power',
              xlim=(data['time'][0], data['time'][-1]), ylim=(0, 1500),
              color='red')
    plt.show()


def plot_temperature(data):
    plot_data(data, x=('time', 'time'), y=('width', 'temperature'),
              color=('blue', 'red'))


def calculate_velocity(time, position):
    velocity = Velocity()
    data = {'speed': [], 'velocity': [], 'running': []}
    for k in range(len(position)):
        speed, vel = velocity.instantaneous(time[k], np.array(position[k]))
        data['speed'].append(speed)
        data['velocity'].append(vel)
        data['running'].append(speed > 0.0005)
    return data


def calculate_geometry(frames, thr=200):
    geometry = Geometry(thr)
    ellipses = [geometry.find_geometry(frame) for frame in frames]
    data = {'x': [], 'y': [], 'height': [], 'width': [], 'angle': []}
    data['x'] = np.array([ellipse[0][0] for ellipse in ellipses])
    data['y'] = np.array([ellipse[0][1] for ellipse in ellipses])
    data['height'] = np.array([ellipse[1][0] for ellipse in ellipses])
    data['width'] = np.array([ellipse[1][1] for ellipse in ellipses])
    data['angle'] = np.array([ellipse[2] for ellipse in ellipses])
    return data


def find_tracks(tachyon, meas='minor_axis'):
    tachyonw = tachyon[tachyon[meas].notnull()]
    laser = np.array(tachyonw[meas] > 0)
    lasernr = np.append(np.bitwise_not(laser[0]), np.bitwise_not(laser[:-1]))
    lasernl = np.append(np.bitwise_not(laser[1:]), np.bitwise_not(laser[-1]))
    laser_on = np.bitwise_and(laser, lasernr)
    laser_off = np.bitwise_and(laser, lasernl)
    laser_on_idx = tachyonw.index[laser_on]
    laser_off_idx = tachyonw.index[laser_off]
    tracks = []
    for k in range(len(laser_on_idx)):
        tracks.append([laser_on_idx[k], laser_off_idx[k]])
    # plt.figure()
    # plt.plot(laser_on)
    # plt.plot(laser_off)
    # plt.show()
    return tracks


def find_data(tachyon, tracks):
    idx0, idx1 = tracks[0][0], tracks[-1][1]
    time0 = tachyon['time'][idx0]
    time1 = tachyon['time'][idx1]
    idx = tachyon.index[tachyon.time < time0-1]
    idx0 = idx[-1]
    idx = tachyon.index[tachyon.time > time1+1]
    idx1 = idx[0]
    print idx0, idx1
    if idx0 < 0:
        idx0 = 0
    if idx1 > len(tachyon):
        idx1 = len(tachyon)
    return idx0, idx1



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


# TODO: Move to ellipse calculation
def resize_ellipse(scale, img, ellipse):
    img = cv2.resize(img, (scale*32, scale*32))
    ((x, y), (w, l), a) = ellipse
    ellipse = ((scale*x, scale*y), (scale*w, scale*l), a)
    return img, ellipse


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
        robot = append_data(robot, velocity)
        plot_speed(robot)

    if 'tachyon' in data.keys():
        tachyon = data['tachyon']
        tachyon = tachyon[tachyon.frame.notnull()]

        if 'minor_axis' in tachyon.columns:
            tracks = find_tracks(tachyon, meas='minor_axis')
            track0 = tracks[0] # first track (laser on, laser off)
            frames = read_frames(tachyon.frame[track0[0]:track0[1]])
            geometry = calculate_geometry(frames, thr=200)
            plot_geometry(geometry)

            plot_frame(tachyon.frame[track0[0] + 100], 200)

            if 'temperature' in tachyon.columns:
                plot_temperature(tachyon)

            # N = len(measures['power'])
            # print measures.loc[N-10:N-1]['time']
            # time = np.array(measures['time'])
            # print np.isclose(time[1:], time[:-1], atol=1e-03, rtol=0)

            if 'power' in tachyon.columns:
                tachyonp = tachyon[tachyon.power.notnull()]
                plot_power(tachyonp)
        else:
            frames = read_frames(tachyon.frame)
            geometry = calculate_geometry(frames, thr=200)
            tachyon = append_data(tachyon, geometry)
            tracks = find_tracks(tachyon, meas='width')
            print 'Tracks:', tracks
            idxs = find_data(tachyon, tracks)
            print 'Data from %i to %i.' % idxs
            plot_geometry(tachyon.loc[idxs[0]:idxs[1]])

            if 'temperature' in tachyon.columns:
                plot_temperature(tachyon)
