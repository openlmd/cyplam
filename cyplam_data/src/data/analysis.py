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
        if laser_off_idx[k] - laser_on_idx[k] > 10:
            tracks.append([laser_on_idx[k], laser_off_idx[k]])
    # plt.figure()
    # plt.plot(laser_on)
    # plt.plot(laser_off)
    # plt.show()
    return tracks


def tracks_times(tachyon, tracks):
    times = []
    for track in tracks:
        times.append([tachyon['time'][track[0]], tachyon['time'][track[1]]])
    return times


def find_tachyon_data(tachyon, tracks):
    idx0, idx1 = tracks[0][0], tracks[-1][1]
    time0 = tachyon['time'][idx0]
    time1 = tachyon['time'][idx1]
    idx = tachyon.index[tachyon.time < time0-1]
    idx0 = idx[-1]
    idx = tachyon.index[tachyon.time > time1+1]
    idx1 = idx[0]
    if idx0 < 0:
        idx0 = 0
    if idx1 > len(tachyon):
        idx1 = len(tachyon)
    print 'Data from %i to %i.' % (idx0, idx1)
    return idx0, idx1


def find_data_tracks(data, times, offset=0.5):
    if type(times[0]) is list:
        time0 = times[0][0]
        time1 = times[-1][1]
    else:
        time0 = times[0]
        time1 = times[1]
    idx = data.index[data.time < time0-offset]
    idx0 = idx[-1]
    idx = data.index[data.time > time1+offset]
    idx1 = idx[0]
    if idx0 < 0:
        idx0 = 0
    if idx1 > len(data):
        idx1 = len(data)
    return data.loc[idx0:idx1]


def max_evolution(tachyon):
    maxims = []
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        maxims.append(img.max())
    maxims = np.array(maxims)
    print maxims.max()
    plot_back(maxims, np.array(tachyon.temperature))


def center_evolution(tachyon):
    back = []
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        back.append(np.mean(img[15:18, 11:13]))
    back = np.array(back)
    print back.max()
    plot_back(back, np.array(tachyon.temperature))


def back_evolution(tachyon):
    back = []
    for frame in tachyon.frame:
        img = deserialize_frame(frame)
        back.append(np.mean(img[25:27, 25:27]))
    back = np.array(back)
    print back.max()
    plot_back(back, np.array(tachyon.temperature))


def plot_back(back, temp):
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
            tachyon = tachyon[tachyon.minor_axis.notnull()]
            plot_data(tachyon, ['time'], ['minor_axis'], ['blue'])

            tracks = find_tracks(tachyon, meas='minor_axis')  # first track (laser on, laser off)
            times = tracks_times(tachyon, tracks)

            frames = read_frames(tachyon.frame.loc[tracks[0][0]:tracks[-1][1]])
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
            times = tracks_times(tachyon, tracks)
            print 'Tracks times:', times

            plot_geometry(find_data_tracks(tachyon, times, offset=0.1))

    if 'camera' in data.keys():
        camera = data['camera']
        print 'Camera length:', len(camera)

        cframes = read_frames(find_data_tracks(camera, times, offset=0).frame)
        plot_frames(cframes)
