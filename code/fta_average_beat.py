import argparse
import json
import shutil

import matplotlib
import pandas as pd

matplotlib.use("agg")
import numpy as np
import os
import matplotlib.pyplot as plt
from hb_utils import interpolate_beat

class BeatSpreader():

    def __init__(self, dimension):
        self.dim = dimension
        return

    def spread(self, beats, beat_det_names, interp_dim):

        beat_set = []
        for beat_det_name in beat_det_names:
            for beat_counter in beats[beat_det_name].keys():
                signal = np.array(beats[beat_det_name][beat_counter])
                # s_min = signal.min()
                signal -= signal.min()
                signal /= signal.max()
                x, y = interpolate_beat(signal, interp_dim)
                beat_set.append(list(y))

        return beat_set


def delete_all_subdirs(base_dir):
    for subdir in os.listdir(base_dir):
        subd_fpath = os.path.join(base_dir, subdir)
        if os.path.isdir(subd_fpath):
            shutil.rmtree(subd_fpath, ignore_errors=True)


def visualize_signal(signals, labels, output_fname, title=""):
    plt.figure()
    plt.title(title)
    for i, signal in enumerate(signals):
        # for visualization normalize signal
        to_plot = (signal - signal.mean()) / signal.std()
        plt.plot(range(signal.shape[0]), to_plot, label=labels[i])
    plt.grid(linestyle='dashed', )
    plt.legend(loc="lower right")
    print(output_fname)
    plt.savefig(output_fname, bbox_inches="tight")
    plt.close()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey it's me")
    parser.add_argument("-f", "--filename", help="Feed me if you want to visualize single file")
    parser.add_argument("-o", "--output_folder", help="Directory where to store filtered csvs",
                        default="/home/data/beat_visuals/")
    parser.add_argument('-b', "--beat_detection_name", help="Beat detection method name", default="hb_argrelmin")
    parser.add_argument("-s", "--display", action='store_true', default=False)
    parser.add_argument("-r", "--force_redo", action='store_true', default=False)
    args = parser.parse_args()

    beat_spreader = BeatSpreader(dimension=1000)

    peaks_folders = [
        ("/home/data/beats/", "NOFTA")
    ]

    i_dim = 480

    os.makedirs(args.output_folder, exist_ok=True)

    for peak_setting in peaks_folders:
        pf, setting = peak_setting
        if args.force_redo:
            delete_all_subdirs(args.output_folder)

        users = os.listdir(pf)
        users = list(filter(lambda x: os.path.isdir(os.path.join(pf, x)), users))

        all_user_beats = {}
        big_beats_arr = np.zeros(shape=(0, i_dim))
        labels = []
        for i, user in enumerate(sorted(users)):
            # get all files
            user_folder = os.path.join(pf, user)
            user_files = os.listdir(user_folder)
            user_files = filter(lambda x: os.path.isfile(os.path.join(user_folder, x)), user_files)

            # only retain allowed formats
            user_files = filter(lambda x: x.split(".")[-1] == "json", user_files)

            user_beats = []
            for file in sorted(user_files):
                filepath = os.path.join(user_folder, file)
                fname = file.split(".")[0]
                extracted_beats = json.load(open(filepath, "r"))
                beat_set = beat_spreader.spread(extracted_beats, [args.beat_detection_name], interp_dim=480)
                user_beats.extend(beat_set)

            all_beats = np.array(user_beats)

            average_beat = np.mean(all_beats, axis=0)
            all_user_beats[user] = average_beat
            big_beats_arr = np.vstack((big_beats_arr, average_beat.reshape(1, -1)))
            labels.append(user)

        df = pd.DataFrame(big_beats_arr, columns=list(range(i_dim)))
        df["label"] = labels

        df.to_csv("/home/data/beat_visuals/%s.csv" % setting, index=False, float_format="%.4f")

        visualize_signal([all_user_beats[x] for x in users], users, os.path.join(args.output_folder, setting + ".pdf"))
