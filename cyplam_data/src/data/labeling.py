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
    from defects import defects

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data', type=str, default=None, help='directory data')
    args = parser.parse_args()

    dirname = args.data

    if dirname is None:
        home = os.path.expanduser('~')
        dirnames = [os.path.join(home, './data/data_set23/20160922_1p_oven'),
                    os.path.join(home, './data/data_set23/20160923_2v_oven'),
                    os.path.join(home, './data/data_set23/20160923_3t_oven'),
                    os.path.join(home, './data/data_set23/20160923_4c_oven'),
                    os.path.join(home, './data/data_set23/20160923_5f_oven')]
    else:
        dirnames = [dirname]

    filenames = get_filenames(dirnames)
    print filenames

    eigenFaces = defects.EigenFaces()
    data, labels = read_datasets(filenames, offset=-0.1)
    frames = read_dataframes(data)
    eigenfaces = eigenFaces.train(np.concatenate(frames))

    # Calculate features and labeling
    offset = 0
    filename = filenames[0]
    data, labels = [], []
    dat, lbls = read_datasets(filenames[0:5], offset=0)
    for k in range(len(dat)):
        frames = analysis.read_frames(dat[k].frame)
        features = {'features': list(eigenFaces.transform(frames))}
        dat[k] = analysis.append_data(dat[k], features)
        data.append(analysis.get_data_array(dat[k], ['features']))
        labels.append(lbls[k])
    features, targets = label_data(data, labels)
    # plot.plot_features_2d(features, targets)

    # plot.plot_frames(np.array(eigenFaces.reconstruct(features)))
    # print 'Total tracks:', len(features)

    clas = defects.Classifier()

    from sklearn.utils import shuffle
    feats = features[targets >= 0]
    targs = targets[targets >= 0]
    # fts, tgs = [], []
    # for tar in np.unique(targs):
    #     fets, tags = shuffle(features[targets == tar], targets[targets == tar],
    #                          random_state=0, n_samples=1000)
    #     fts.append(fets)
    #     tgs.append(tags)
    # feats = np.vstack(fts)
    # targs = np.hstack(tgs)

    labls = [str(label) for label in np.unique(targs)]
    clas.train_test(feats, targs, labls)
    defects.plot.plot_features(feats, targs)

    prediction = [clas.predict(X) for X in features]
    defects.plot.plot_features(features, prediction)

    # data, labels = read_datasets([filenames[0]], offset=0)
    # features = eigenFaces.transform(np.concatenate(data))
    # prediction = [clas.predict(X) for X in features]
    # defects.plot.plot_features(features, prediction)

    dirnames = [os.path.join(home, './data/data_set23/20160922_1p'),
                os.path.join(home, './data/data_set23/20160923_2v'),
                os.path.join(home, './data/data_set23/20160923_3t'),
                os.path.join(home, './data/data_set23/20160923_4c'),
                os.path.join(home, './data/data_set23/20160923_5f'),
                os.path.join(home, './data/data_set23/20160923_6c'),
                os.path.join(home, './data/data_set23/20160922_1p_oven'),
                os.path.join(home, './data/data_set23/20160923_2v_oven'),
                os.path.join(home, './data/data_set23/20160923_3t_oven'),
                os.path.join(home, './data/data_set23/20160923_4c_oven'),
                os.path.join(home, './data/data_set23/20160923_5f_oven'),
                os.path.join(home, './data/data_set23/20160923_6c_oven')]
    filenames = get_filenames(dirnames)

    for n, filename in enumerate(filenames):
        data, labels = read_datasets([filename], offset=0)
        for k, dat in enumerate(data):
            frames = analysis.read_frames(dat.frame)
            features = {'features': list(eigenFaces.transform(frames))}
            dat = analysis.append_data(dat, features)
            features = analysis.get_data_array(dat, ['features'])
            prediction = np.array([clas.predict(X) for X in features])
            score = float(np.sum(prediction))/len(prediction)
            sums = [np.sum(prediction == l) for l in range(5)]
            print 'D', n, 't', k, 'score', score, 'labels', sums
