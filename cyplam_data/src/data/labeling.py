import os
import glob
import numpy as np

import analysis


def read_frames_track(tachyon, track, offset=0):
    data = analysis.find_data_tracks(tachyon, track, offset)
    frames = analysis.read_frames(data.frame)
    return frames


def read_dataset(filename, offset=-0.1):
    tachyon, tracks, labels = analysis.read_tachyon_data(filename)
    #plot.plot_geometry(tachyon)
    data = [analysis.find_data_tracks(
        tachyon, track, offset) for track in tracks]
    return data, labels


def read_dataframes(data):
    return [analysis.read_frames(data[k].frame) for k in range(len(data))]


def read_datasets(filenames, offset=-0.1):
    data, labels = [], []
    for filename in filenames:
        dat, lbls = read_dataset(filename, offset)
        data.extend(dat)
        labels.extend(lbls)
    return data, labels


def label_data(data, labels):
    features, targets = [], []
    for track, dat in enumerate(data):
        features.append(dat)
        targets.append(np.array([labels[track]] * len(dat)))
    features = np.concatenate(features)
    targets = np.concatenate(targets)
    return features, targets


def get_filenames(dirnames):
    filenames = []
    for dirname in dirnames:
        filesnames = sorted(glob.glob(os.path.join(dirname, '*.bag')))
        filenames.append(filesnames[-1])
    return filenames


if __name__ == "__main__":
    import plot
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data', type=str, default=None, help='directory data')
    args = parser.parse_args()

    dirname = args.data

    if dirname is None:
        home = os.path.expanduser('~')
        dirnames = [os.path.join(home, './data/data_nov24/24112016_1v_900'),
                    os.path.join(home, './data/data_nov24/24112016_2v_1000'),
                    os.path.join(home, './data/data_nov24/24112016_3p_900'),
                    os.path.join(home, './data/data_nov24/24112016_4p_1000')]
    else:
        dirnames = [dirname]

    filenames = get_filenames(dirnames)
    data, labels = read_datasets(filenames, offset=-0.1)
    frames = read_dataframes(data)

    dat, lbls = read_datasets([filenames[0]], offset=0)
    for k in range(len(dat)):
        frames = analysis.read_frames(dat[k].frame)
        plot.plot_frames(frames)
