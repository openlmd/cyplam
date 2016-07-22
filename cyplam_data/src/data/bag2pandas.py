import cv2
import rosbag
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from mashes_measures.msg import MsgGeometry
from cladplus_control.msg import MsgPower

from cv_bridge import CvBridge

bridge = CvBridge()


def serialize_frame(frame, encode='.tiff'):
    return cv2.imencode(encode, frame)[1].tostring()


def deserialize_frame(string):
    return cv2.imdecode(np.fromstring(string, np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)


def read_frames(frames):
    return np.array([deserialize_frame(frame) for frame in frames])


def read_topic_list(bag):
    topicList = []
    for topic, msg, t in bag.read_messages():
        if topicList.count(topic) == 0:
            topicList.append(topic)
    return topicList


def read_topic_data(bag, topic):
    data_dic = {}
    for idx, (topic, msg, mt) in enumerate(bag.read_messages(topics=topic)):
        if idx == 0:
            slots = msg.__slots__
            columns = list(slots)
            columns[columns.index('header')] = 'time'
            for column in columns:
                data_dic[column] = []
        for slot in slots:
            if slot == 'header':
                data_dic['time'].append(msg.header.stamp.to_sec())
            else:
                data_dic[slot].append(getattr(msg, slot))
    data = pd.DataFrame(data_dic)
    return data


def read_topic_img(bag, topic):
    times, frames = [], []
    for topic, msg, t in bag.read_messages(topics=topic):
        frame = bridge.imgmsg_to_cv2(msg)
        times.append(msg.header.stamp.to_sec())
        frames.append(serialize_frame(frame))
    images = pd.DataFrame({'time': times, 'frame': frames})
    return images


def merge_topic_data(data1, data2):
    data = pd.merge(data1, data2, how='outer', on='time', sort=True)
    #control = pd.concat([self.geometry, self.control], axis=1)
    return data


if __name__ == "__main__":
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--bag', type=str, default=None, help='input bag file')
    parser.add_argument(
        '-t', '--topics', help='list of topics',  action='store_true')
    args = parser.parse_args()

    filename = args.bag
    if filename is None:
        filenames = sorted(glob.glob('/home/jorge/data/*.bag'))
        filename = filenames[-1]
    filename = '/home/jorge/data/data_20160721-003440.bag'
    print filename

    if args.topics:
        bag = rosbag.Bag(filename)
        topics = read_topic_list(bag)
        bag.close()
        print 'Topics:', topics

    # Get dataframes
    bag = rosbag.Bag(filename)
    images = read_topic_img(bag, '/tachyon/image')
    #geometry = read_topic_data(bag, '/tachyon/geometry')
    #control = read_topic_data(bag, '/control/power')
    camera = read_topic_img(bag, '/camera/image')
    joints = read_topic_data(bag, '/joint_states')
    bag.close()

    # TODO: Read all the topics as dataframes and merge corresponding subtopics
    # /tachyon/image + /tachyon/geometry = /tachyon (dataframe)

    # Measures, merge dataframes example
    #measures = merge_topic_data(images, geometry)
    #measures = merge_topic_data(measures, control)
    #measures = measures.rename(columns={'value': 'power'})
    #measures.columns
    measures = images

    # Save HDF5 file
    images.to_hdf('foo.h5', 'images')
    camera.to_hdf('foo.h5', 'camera')
    joints.to_hdf('foo.h5', 'joints')
    # pd.read_hdf('foo.h5','geometry')

    # <codecell>
    img_str = measures.loc[69]['frame']
    img = deserialize_frame(img_str)
    plt.figure()
    plt.imshow(img, interpolation='none')
    plt.show()

    frames = read_frames(measures[measures['minor_axis'] > 0]['frame'])
    print frames.shape
    plt.figure()
    plt.imshow(frames[50], interpolation='none')
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
    measures.plot(x='time', y='minor_axis')
    plt.subplot(212)
    measures.plot(x='time', y='power')
    plt.show()
    measures

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
    measures.plot(x='time', y='minor_axis', xlim=(time[0], time[-1]), ylim=(0, 5))
    plt.subplot(212)
    measures.plot(x='time', y='power', xlim=(time[0], time[-1]), ylim=(0, 1500))

    # <codecell>
    mean = meas['minor_axis'].mean()
    std = meas['minor_axis'].std()
    print mean, std
