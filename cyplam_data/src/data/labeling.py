import numpy as np

from data import plot
from data import analysis


def read_frames_track(tachyon, track, offset=0):
    data = analysis.find_data_tracks(tachyon, track, offset)
    frames = analysis.read_frames(data.frame)
    return frames


def read_dataset(filename, offset=-0.1):
    tachyon, tracks, labels = analysis.read_tachyon_data(filename)
    #plot.plot_geometry(tachyon)
    data = [analysis.find_data_tracks(
        tachyon, track, offset) for track in tracks]
    frames = [analysis.read_frames(data[k].frame) for k in range(len(data))]
    return frames, labels


def read_datasets(filenames, offset=-0.1):
    data, labels = [], []
    for filename in filenames:
        frames, lbls = read_dataset(filename, offset)
        data.extend(frames)
        labels.extend(lbls)
    data, labels = np.array(data), np.array(labels)
    return data, labels


def label_data(data, labels):
    features, targets = [], []
    for track, dat in enumerate(data):
        features.append(dat)
        targets.append(np.array([labels[track]] * len(dat)))
    features = np.concatenate(features)
    targets = np.concatenate(targets)
    return features, targets


if __name__ == "__main__":
    import os
    import glob
    import argparse
    from defects import defects

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data', type=str, default=None, help='directory data')
    args = parser.parse_args()

    dirname = args.data

    if dirname is None:
        dirnames = ['/home/jorge/data/data_set23/20160922_1p_oven',
                    '/home/jorge/data/data_set23/20160923_2v_oven',
                    '/home/jorge/data/data_set23/20160923_3t_oven',
                    '/home/jorge/data/data_set23/20160923_4c_oven',
                    '/home/jorge/data/data_set23/20160923_5f_oven',
                    '/home/jorge/data/data_set23/20160923_6c_oven']
    else:
        dirnames = [dirname]

    filenames = []
    for dirname in dirnames:
        filesnames = sorted(glob.glob(os.path.join(dirname, '*.bag')))
        filenames.append(filesnames[-1])
    print filenames

    eigenFaces = defects.EigenFaces()
    data, labels = read_datasets(filenames, offset=-0.1)
    eigenfaces = eigenFaces.train(np.concatenate(data))
    plot.plot_frames(eigenfaces)

    # Calculate features and labeling
    offset = 0
    filename = filenames[0]
    data, labels = [], []
    for filename in filenames[0:2]:
        tachyon, tracks, lbls = analysis.read_tachyon_data(filename)
        frames = analysis.read_frames(tachyon.frame)
        features = eigenFaces.transform(frames)
        tachyon = analysis.append_data(tachyon, {'features': list(features)})
        dat = [analysis.find_data_tracks(tachyon, track, 0) for track in tracks]
        #frames = [analysis.read_frames(data[k].frame) for k in range(len(data))]
        [data.append(analysis.get_data_array(d, ['width', 'features'])) for d in dat]
        labels.extend(lbls)
    features, targets = label_data(data, labels)
    # plot.plot_features_2d(features, targets)

    # plot.plot_frames(np.array(eigenFaces.reconstruct(features)))
    # print 'Total tracks:', len(features)

    clas = defects.Classifier()

    feats = features[targets >= 0]
    targs = targets[targets >= 0]
    labls = [str(label) for label in np.unique(targs)]
    clas.train_test(feats, targs, labls)
    defects.plot.plot_features(feats, targs)

    prediction = [clas.predict(X) for X in features]
    defects.plot.plot_features(features, prediction)

    # data, labels = read_datasets([filenames[0]], offset=0)
    # features = eigenFaces.transform(np.concatenate(data))
    # prediction = [clas.predict(X) for X in features]
    # defects.plot.plot_features(features, prediction)

    for n, filename in enumerate(filenames):
        data, labels = read_datasets([filename], offset=0)
        for k, dat in enumerate(data):
            features = eigenFaces.transform(dat)
            prediction = np.array([clas.predict(X) for X in features])
            #defects.plot.plot_features(features, prediction)
            score = float(np.sum(prediction))/len(prediction)
            print n, 'score', k, score, np.sum(prediction==0), np.sum(prediction==1), np.sum(prediction==2), np.sum(prediction==3), np.sum(prediction==4)

    # features, labels = [], []
    # for track in tracks:
    #     data = analysis.find_data_tracks(tachyon, track, 0)
    #     feats = analysis.get_data_array(data, ['maximum', 'features'])
    #     labls = np.array(data.label)
    #     features.extend(feats)
    #     labels.extend(labls)
    # features = np.array(features)
    # labels = np.array(labels)
