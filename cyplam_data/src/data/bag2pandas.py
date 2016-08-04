import cv2
import rosbag
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from kinematics import Kinematics

from cv_bridge import CvBridge

bridge = CvBridge()


def serialize_frame(frame, encode='.tiff'):
    return cv2.imencode(encode, frame)[1].tostring()


def deserialize_frame(string):
    return cv2.imdecode(
        np.fromstring(string, np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)


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


def transform_joints(joints):
    kdl = Kinematics()
    kdl.set_kinematics('base_link', 'tcp0')
    poses = [kdl.get_pose(jnts) for jnts in joints]
    return poses


def read_topic_joints(bag, topic='/joint_states'):
    data_dic = {}
    for idx, (topic, msg, mt) in enumerate(bag.read_messages(topics=topic)):
        if idx == 0:
            slots = msg.__slots__
            columns = ['time', 'position']
            for column in columns:
                data_dic[column] = []
        for slot in slots:
            if slot == 'header':
                data_dic['time'].append(msg.header.stamp.to_sec())
            if slot == 'position':
                data_dic[slot].append(getattr(msg, slot))
    data_dic['position'] = transform_joints(data_dic['position'])
    data = pd.DataFrame(data_dic)
    return data


def merge_topic_data(data1, data2):
    data = pd.merge(data1, data2, how='outer', on='time', sort=True)
    #control = pd.concat([self.geometry, self.control], axis=1)
    return data


def read_topics(topics):
    data = {}
    for topic in topics:
        names = topic.split('/')[1:]
        if names[-1] == 'image':
            dat = read_topic_img(bag, topic)
            if names[-2] in data.keys():
                data[names[-2]] = merge_topic_data(data[names[-2]], dat)
            else:
                data[names[-2]] = dat
        elif names[-1] == 'joint_states':
            data['robot'] = read_topic_joints(bag, topic)
        elif len(names) > 1:
            dat = read_topic_data(bag, topic)
            if names[-2] in data.keys():
                data[names[-2]] = merge_topic_data(data[names[-2]], dat)
            else:
                data[names[-2]] = dat
    return data


if __name__ == "__main__":
    import os
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
        filename = '/home/jorge/data/20_2000.bag'
    print filename

    if args.topics:
        bag = rosbag.Bag(filename)
        topics = read_topic_list(bag)
        bag.close()
        print 'Topics:', topics

    # Get dataframes
    bag = rosbag.Bag(filename)
    topics = read_topic_list(bag)
    data = read_topics(topics)
    #geometry = read_topic_data(bag, '/tachyon/geometry')
    #control = read_topic_data(bag, '/control/power')
    bag.close()

    print 'Data:', data.keys()

    if 'control' in data.keys() and 'tachyon' in data.keys():
        data['tachyon'] = merge_topic_data(data['tachyon'], data['control'])
        data['tachyon'] = data['tachyon'].rename(columns={'value': 'power'})
        del data['control']

    # Measures, merge dataframes example
    #measures = merge_topic_data(geometry, control)
    #measures = measures.rename(columns={'value': 'power'})
    #measures.columns
    measures = data['tachyon']

    # TODO: Read all the topics as dataframes and merge corresponding subtopics
    # /tachyon/image + /tachyon/geometry = /tachyon (dataframe)
    # merge_dataframes

    # Save HDF5 file
    filename += '.h5'
    if os.path.isfile(filename):
        os.remove(filename)
    for name, dat in data.iteritems():
        dat.to_hdf(filename, name)
    # pd.read_hdf(filename,'geometry')

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
    measures.plot(x='time', y='minor_axis', xlim=(time[0], time[-1]), ylim=(0, 5), color='blue')
    plt.subplot(212)
    measures.plot(x='time', y='power', xlim=(time[0], time[-1]), ylim=(0, 1500), color='red')
    plt.show()

    # <codecell>
    mean = meas['minor_axis'].mean()
    std = meas['minor_axis'].std()
    print mean, std
