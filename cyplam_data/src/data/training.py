import os
import numpy as np

import analysis
import labeling


if __name__ == "__main__":
    import plot
    from defects import defects

    home = os.path.expanduser('~')
    dirnames = [os.path.join(home, './data/data_set23/20160922_1p_oven'),
                os.path.join(home, './data/data_set23/20160923_2v_oven'),
                os.path.join(home, './data/data_set23/20160923_5f_oven')]

    filenames = labeling.get_filenames(dirnames)
    print filenames

    eigenFaces = defects.EigenFaces()
    data, labels = labeling.read_datasets(filenames, offset=-0.1)
    frames = labeling.read_dataframes(data)
    eigenfaces = eigenFaces.train(np.concatenate(frames))
    plot.plot_frames(eigenfaces)

    # Calculate features and labeling
    data, labels = [], []
    dat, lbls = labeling.read_datasets([filenames[1]], offset=0)
    for k in range(len(dat)):
        frames = analysis.read_frames(dat[k].frame)
        features = {'features': list(eigenFaces.transform(frames))}
        dat[k] = analysis.append_data(dat[k], features)
        data.append(analysis.get_data_array(dat[k], ['features']))
        labels.append(lbls[k])
    features, targets = labeling.label_data(data, labels)
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
                os.path.join(home, './data/data_set23/20160923_5f'),
                os.path.join(home, './data/data_set23/20160922_1p_oven'),
                os.path.join(home, './data/data_set23/20160923_2v_oven'),
                os.path.join(home, './data/data_set23/20160923_5f_oven')]
    filenames = labeling.get_filenames(dirnames)

    for n, filename in enumerate(filenames):
        data, labels = labeling.read_datasets([filename], offset=0)
        for k, dat in enumerate(data):
            frames = analysis.read_frames(dat.frame)
            features = {'features': list(eigenFaces.transform(frames))}
            dat = analysis.append_data(dat, features)
            features = analysis.get_data_array(dat, ['features'])
            prediction = np.array([clas.predict(X) for X in features])
            score = float(np.sum(prediction))/len(prediction)
            sums = [np.sum(prediction == l) for l in range(5)]
            print 'D', n, 't', k, 'score', score, 'labels', sums
