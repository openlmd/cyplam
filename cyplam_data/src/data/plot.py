import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import analysis


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
    img = analysis.deserialize_frame(frame)
    plot_image(img)
    plot_image3d(img)  # Show image as 3D surface
    plot_histogram(img)


def plot_frames(frames, N=30, limits=None):
    n = len(frames)
    if n > N:
        step = n / (N - 1)
        off = (n - step * N) / 2 + 1
        idx = range(off, n-off, step)
        frames = frames[idx]
        n = len(frames)
    c = int(np.ceil(np.sqrt(n)))
    r = int(np.ceil(1. / c * n))
    gs = gridspec.GridSpec(r, c)
    gs.update(wspace=0.025, hspace=0.05)
    plt.figure()
    for k in range(n):
        ax = plt.subplot(gs[k])
        if limits is None:
            plt.imshow(frames[k], interpolation='none')
        else:
            plt.imshow(
                frames[k], interpolation='none', vmin=limits[0], vmax=limits[1])
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


def plot_digital(data):
    plot_data(data, ['time'], ['digital_level'], ['blue'])
