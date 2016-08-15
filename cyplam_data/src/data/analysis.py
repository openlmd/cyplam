import cv2
import numpy as np
import pandas as pd


def serialize_frame(frame, encode='.tiff'):
    return cv2.imencode(encode, frame)[1].tostring()


def deserialize_frame(string):
    return cv2.imdecode(
        np.fromstring(string, np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)


def read_frames(frames):
    return np.array([deserialize_frame(frame) for frame in frames])


def write_hdf5(filename, data, keys=None):
    store = pd.HDFStore(filename)
    if keys is None:
        keys = data.keys()
    for key in keys:
        store.put(key, data[key])
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


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', type=str, default=None, help='input hdf5 file')
    parser.add_argument(
        '-t', '--topics', help='list of topics',  action='store_true')
    args = parser.parse_args()

    filename = args.file
    data = read_hdf5(filename)

    # Show image
    img = deserialize_frame(data['tachyon'].frame[69])
    plt.figure()
    plt.imshow(img, interpolation='none')
    plt.show()

    # Show image as 3D surface
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    # create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap='jet', linewidth=0)
    plt.show()

    measures = data['tachyon'][data['tachyon'].frame.notnull()]
    frames = read_frames(measures[measures['minor_axis'] > 0]['frame'])
    print frames.shape
    plt.figure()
    plt.imshow(frames[50], interpolation='none')
    plt.figure()
    plt.hist(img.flatten(), 100, range=(1, 1024), fc='k', ec='k')
    plt.show()

    # <codecell>
    print 'Len power measures:', len(measures['power'])
    print 'Len minor_axis measures:', len(measures['minor_axis'])
    N = len(measures['power'])
    print measures.loc[N-10:N-1]['time']
    time = np.array(measures['time'])
    print np.isclose(time[1:], time[:-1], atol=1e-03, rtol=0)

    # <codecell>
    measures['time'] = measures['time'] - measures['time'][0]
    measures = measures[measures['power'].notnull()]
    plt.subplot(211)
    measures.plot(x='time', y='minor_axis', color='blue')
    plt.subplot(212)
    measures.plot(x='time', y='power', color='red')
    plt.show()

    # <codecell>
    idx = measures.index[measures['minor_axis'] > 0]
    idx0, idx1 = idx[0], idx[-1]
    time0 = measures['time'][idx0]
    time1 = measures['time'][idx1]
    print idx0, idx1, time0, time1
    idx = measures.index[measures['time'] < time0-1]
    idx0 = idx[-1]
    idx = measures.index[measures['time'] > time1+1]
    idx1 = idx[0]
    print idx0, idx1
    if idx0 < 0:
        idx0 = 0
    if idx1 > N:
        idx1 = N
    meas = measures.loc[idx0:idx1]
    time = np.array(meas['time'])
    plt.subplot(211)
    measures.plot(x='time', y='minor_axis',
                  xlim=(time[0], time[-1]), ylim=(0, 5), color='blue')
    plt.subplot(212)
    measures.plot(x='time', y='power',
                  xlim=(time[0], time[-1]), ylim=(0, 1500), color='red')
    plt.show()
