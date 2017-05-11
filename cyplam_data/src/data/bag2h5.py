import gc
import cv2
import rosbag
import numpy as np
import pandas as pd

from kinematics import Kinematics

from cv_bridge import CvBridge


def serialize_frame(frame, encode='*.png'):
    return cv2.imencode(encode, frame)[1].tostring()


def deserialize_frame(string):
    return cv2.imdecode(
        np.fromstring(string, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)


def read_topic_list(bag):
    topicList = []
    for topic, msg, t in bag.read_messages():
        if topicList.count(topic) == 0:
            topicList.append(topic)
    return topicList


def transform_joints(joints):
    kdl = Kinematics()
    kdl.set_kinematics('base_link', 'tcp0')
    poses = [kdl.get_pose(jnts[:6]) for jnts in joints]
    return poses


def merge_topic_data(data1, data2):
    data = pd.merge(data1, data2, how='outer', on='time', sort=True)
    return data


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
    bridge = CvBridge()
    times, frames = [], []
    for topic, msg, t in bag.read_messages(topics=topic):
        frame = bridge.imgmsg_to_cv2(msg)
        times.append(msg.header.stamp.to_sec())
        frames.append(serialize_frame(frame))
    images = pd.DataFrame({'time': times, 'frame': frames})
    return images

def read_predict_power(bag, topic):
    dat_pre= {}
    dat_pre= read_topic_data(bag, topic)
    dat_pre= dat_pre.rename(columns={'value': 'power'})
    return dat_pre


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
    poses = transform_joints(data_dic['position'])
    data_dic['position'] = [pose[0] for pose in poses]
    data_dic['orientation'] = [pose[1] for pose in poses]
    data = pd.DataFrame(data_dic)
    return data


def read_topics(bag, topics):
    data = {}
    for topic in topics:
        names = filter(None, topic.split('/'))
        if names[-1] == 'image':
            dat = read_topic_img(bag, topic)
            if names[-2] in data.keys():
                data[names[-2]] = merge_topic_data(data[names[-2]], dat)
            else:
                data[names[-2]] = dat
        elif names[-1] == 'joint_states':
            data['robot'] = read_topic_joints(bag, topic)
        elif names[-1] == 'power' and names[-2] == 'predict':
            dat= read_predict_power(bag, topic)
            if 'predict' in data.keys():
                data['predict'] = merge_topic_data(data['predict'], dat)
            else:
                data['predict'] = dat
        elif len(names) > 1:
            dat = read_topic_data(bag, topic)
            if names[-2] in data.keys():
                data[names[-2]] = merge_topic_data(data[names[-2]], dat)
            else:
                data[names[-2]] = dat
    return data


def read_bag_data(filename, topics=None):
    bag = rosbag.Bag(filename)
    if topics is None:
        topics = read_topic_list(bag)
    data = read_topics(bag, topics)
    bag.close()
    return data


def write_hdf5(filename, data, keys=None):
    store = pd.HDFStore(filename, complevel=9, complib='blosc')
    #store = pd.HDFStore(filename, complevel=1, complib="zlib")
    if keys is None:
        keys = data.keys()
    for key in keys:
        store.put(key, data[key], format='fixed', append=False)
        #store.put(key, data[key], format='table', append=False)
        #gc.collect()
        #print gc.collect()
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
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--bag', type=str, default=None, help='bag filename')
    parser.add_argument(
        '-t', '--topics', type=str, default=None, nargs='+', help='topics')
    parser.add_argument(
        '-i', '--info', help='list of topics',  action='store_true')
    args = parser.parse_args()

    filename = args.bag
    topics = args.topics

    if args.info:
        bag = rosbag.Bag(filename)
        topics = read_topic_list(bag)
        bag.close()
        print 'Topics:', topics

    data = read_bag_data(filename, topics)
    #geometry = read_bag_data(filename, ['/tachyon/geometry'])

    # Merges control data in tachyon dataframe and rename value to power
    if 'control' in data.keys() and 'tachyon' in data.keys():
        data['tachyon'] = merge_topic_data(data['tachyon'], data['control'])
        data['tachyon'] = data['tachyon'].rename(columns={'value': 'power'})
        del data['control']

    filename = os.path.splitext(filename)[0] + '.h5'
    write_hdf5(filename, data)
    #data = read_hdf5(filename)
