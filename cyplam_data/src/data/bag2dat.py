from data import plot
from data import bag2h5
from data import analysis

from tachyon.nitdat import NitDat


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', type=str, default=None, help='bag filename')
    args = parser.parse_args()

    filename = args.file
    data = bag2h5.read_bag_data(filename, ['/tachyon/image'])
    tachyon = data['tachyon']

    frames = analysis.read_frames(tachyon.frame)
    geometry = analysis.calculate_geometry(frames, thr=150)
    tachyon = analysis.append_data(tachyon, geometry)
    tracks = analysis.find_tracks(tachyon)
    print 'Tracks:', tracks

    tachyon = analysis.find_data_tracks(tachyon, tracks, offset=1)
    frames = analysis.read_frames(tachyon.frame)
    plot.plot_geometry(tachyon)

    name, ext = os.path.splitext(filename)
    filename = name + '.dat'

    dat = NitDat()
    dat.write_frames(filename, frames)
    plot.plot_frames(frames)
