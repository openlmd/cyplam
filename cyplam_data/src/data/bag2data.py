import os
import cv2
import shutil
import rosbag

from cv_bridge import CvBridge


bridge = CvBridge()


def extract(filename):
    path = os.getcwd()
    name = os.path.splitext(filename)[0]
    basename = os.path.basename(name)
    directory = os.path.join(path, basename)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    bag = rosbag.Bag(filename)
    topicList = readBagTopicList(bag)
    while True:
        if len(topicList) == 0:
            print "No topics in list. Exiting"
            break
        selection = menu(topicList)
        if selection == -92:
            print "[OK] Printing them all"
            for topic in topicList:
                print 'Topic:', topic
                name = topic.split('/')
                path_dir = os.path.join(path, basename, name[1])
                os.mkdir(path_dir)
                extract_data(bag, topic, path_dir)
            break
        elif selection == -45:
            break
        else:
            topic = topicList[selection]
            name = topic.split('/')
            path_dir = os.path.join(path, basename, name[1])
            os.mkdir(path_dir)
            extract_data(bag, topic, path_dir)
            topicList.remove(topicList[selection])
    bag.close()


def extract_data(bag, topic, path_dir):
    filename = os.path.join(path_dir, topic.split('/')[-1] + '.txt')
    print "[OK] Printing %s" % topic
    print "[OK] Output file will be called %s." % filename
    outputFh = open(filename, "w")
    for topic, msg, t in bag.read_messages(topics=topic):
        try:
            t = msg.header.stamp
            frame = bridge.imgmsg_to_cv2(msg)
            filename = os.path.join(path_dir, str(t) + '.png')
            print 'Image:', filename
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(filename, frame)
            cv2.imshow('image', frame)
            cv2.waitKey(1)
        except:
            data = []
            for slot in msg.__slots__:
                if slot == 'header':
                    data.append(msg.header.stamp)
                else:
                    data.append(getattr(msg, slot))
            data = ', '.join([str(dat) for dat in data])
            outputFh.write(data + '\n')

    outputFh.close()
    print "[OK] DONE"


def menu(topicList):
    i = 0
    for topic in topicList:
        print '[{0}] {1}'.format(i, topic)
        i = i+1
    if len(topicList) > 1:
        print '[{0}] Extract all'.format(len(topicList))
        print '[{0}] Exit'.format(len(topicList) + 1)
    else:
        print '[{0}] Exit'.format(len(topicList))
    while True:
        print 'Enter a topic number to extract raw data from:'
        selection = raw_input('>>>')
        if int(selection) == len(topicList):
            return -92  # print all
        elif int(selection) == (len(topicList) + 1):
            return -45  # exit
        elif (int(selection) < len(topicList)) and (int(selection) >= 0):
            return int(selection)
        else:
            print "[ERROR] Invalid input"


def readBagTopicList(bag):
    print "[OK] Reading topics in this bag. Can take a while.."
    topicList = []
    for topic, msg, t in bag.read_messages():
        if topicList.count(topic) == 0:
            topicList.append(topic)
    print '{0} topics found:'.format(len(topicList))
    return topicList


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bag', type=str,
                        default='13.bag',
                        help='path to input bag file')
    args = parser.parse_args()

    filename = args.bag

    extract(filename)
