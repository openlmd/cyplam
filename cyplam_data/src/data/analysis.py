import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def plot_frame(frame, thr=150):
    img = deserialize_frame(frame)
    geometry = Geometry(thr)
    ellipse = geometry.find_geometry(img)
    plot_image(geometry.draw_geometry(img.copy(), ellipse))
    plot_image3d(img)  # Show image as 3D surface
    plot_histogram(img)


def plot_frames(frames, N=30):
    n = len(frames)
    if n > N:
        step = n / (N - 1)
        off = (n - step * N) / 2 + 1
        idx = range(off, n-off, step)
        print len(idx)
        frames = frames[idx]
        n = len(frames)
    c = int(np.ceil(np.sqrt(n)))
    r = int(np.ceil(1. / c * n))
    gs = gridspec.GridSpec(r, c)
    gs.update(wspace=0.025, hspace=0.05)
    plt.figure()
    for k in range(n):
        ax = plt.subplot(gs[k])
        plt.imshow(frames[k], interpolation='none')
        #plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_aspect('equal')
    plt.show()


def plot_data(data, x, y, color):
    rows = len(y)
    plt.figure()
    for k in range(rows):
        plt.subplot(rows, 1, k+1)
        plt.plot(data[x[k]], data[y[k]], color=color[k])
        plt.xlim(data[x[k]].min(), data[x[k]].max())
        plt.ylabel(y[k])
    plt.show()


def plot_speed(data):
    plot_data(data, x=('time', 'time'), y=('speed', 'running'),
              color=('blue', 'red'))


def plot_geometry(data):
    plot_data(data, x=('time', 'time'), y=('width', 'height'),
              color=('blue', 'green'))


def plot_power(data):
    plot_data(data, x=('time', 'time'), y=('minor_axis', 'power'),
              color=('blue', 'red'))


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


def find_tracks(tachyon, meas='width'):
    tachyonw = tachyon[tachyon[meas].notnull()]
    laser = np.array(tachyonw[meas] > 0)
    lasernr = np.append(np.bitwise_not(laser[0]), np.bitwise_not(laser[:-1]))
    lasernl = np.append(np.bitwise_not(laser[1:]), np.bitwise_not(laser[-1]))
    laser_on = np.bitwise_and(laser, lasernr)
    laser_off = np.bitwise_and(laser, lasernl)
    lon_idx = tachyonw.index[laser_on]
    loff_idx = tachyonw.index[laser_off]
    tracks = []
    for k in range(len(lon_idx)):
        if loff_idx[k] - lon_idx[k] > 30:
            tracks.append([tachyon['time'][lon_idx[k]],
                           tachyon['time'][loff_idx[k]]])
    return tracks


def find_data_tracks(data, tracks, offset=1.0):
    if type(tracks[0]) is list:
        time0 = tracks[0][0]
        time1 = tracks[-1][1]
    else:
        time0 = tracks[0]
        time1 = tracks[1]
    idx0 = data.index[data.time < time0-offset][-1]
    idx1 = data.index[data.time > time1+offset][0]
    if idx0 < 0:
        idx0 = 0
    if idx1 > len(data):
        idx1 = len(data)
    return data.loc[idx0:idx1]


def calculate_back(tachyon):
    back = {'digital_level': []}
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        back['digital_level'].append(np.mean(img[25:27, 25:27]))
    print max(back['digital_level'])
    return back


def calculate_clad(tachyon):
    data = {'digital_level': []}
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        data['digital_level'].append(np.mean(img[15:18, 11:13]))
    print max(data['digital_level'])
    return data


def calculate_maximun(tachyon):
    data = {'digital_level': []}
    i=0
    for frame in tachyon.frame:
        img = deserialize_frame(frame)

        data['digital_level'].append(img.max())
        if img.max()>400:
            print i
        i = i + 1
    print max(data['digital_level'])
    return data


def plot_digital(tachyon, data):
    tachyon = append_data(tachyon, data)
    plot_data(tachyon, ['time'], ['digital_level'], ['blue'])


# TODO: Move to ellipse calculation
def resize_ellipse(scale, img, ellipse):
    img = cv2.resize(img, (scale*32, scale*32))
    ((x, y), (w, l), a) = ellipse
    ellipse = ((scale*x, scale*y), (scale*w, scale*l), a)
    return img, ellipse


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--bag', type=str, default=None, help='bag filename')
    parser.add_argument(
        '-f', '--file', type=str, default=None, help='hdf5 filename')
    parser.add_argument(
        '-t', '--tables', type=str, default=None, nargs='+', help='tables names')
    args = parser.parse_args()

    if args.bag:
        import bag2h5
        data = bag2h5.read_bag_data(args.bag)
        name, ext = os.path.splitext(args.bag)
        filename = name + '.h5'
        bag2h5.write_hdf5(filename, data)

    if args.file:
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
            tachyon = tachyon[tachyon.minor_axis.notnull()]
            plot_data(tachyon, ['time'], ['minor_axis'], ['blue'])

            tracks = find_tracks(tachyon, meas='minor_axis')  # first track (laser on, laser off)
            frames = read_frames(find_data_tracks(tachyon, tracks, offset=0).frame)
            plot_frames(frames)

            if 'temperature' in tachyon.columns:
                plot_temperature(tachyon)

            if 'power' in tachyon.columns:
                plot_power(tachyon[tachyon.power.notnull()])
        else:
            frames = read_frames(tachyon.frame)
            geometry = calculate_geometry(frames, thr=150)
            tachyon = append_data(tachyon, geometry)
            tracks = find_tracks(tachyon, meas='width')
            print 'Tracks:', tracks

            plot_geometry(find_data_tracks(tachyon, tracks, offset=0.1))

    if 'camera' in data.keys():
        camera = data['camera']
        print 'Camera length:', len(camera)

        cframes = read_frames(find_data_tracks(camera, tracks, offset=0).frame)
        plot_frames(cframes)
