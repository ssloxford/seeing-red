import argparse
import json
import shutil

import matplotlib
import yaml

matplotlib.use("agg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin,argrelmax
import os
import sys
import matplotlib.patches as patches
import hb_utils as hbutils


class SignalFiducialPointsDetector():

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def moving_average(self, x, n):
        return np.convolve(x, np.ones(n), 'same') / n

    def first_minmax_if_exists(self, x, kind):
        if kind == "min":
            mins = argrelmin(x)[0]
            if len(mins)>0:
                return mins[0]
            else:
                return 0
        if kind == "max":
            maxs = argrelmax(x)[0]
            if len(maxs) > 0:
                return maxs[0]
            else:
                return 0

    def hb_fp_detection_20191217(self, beats, **kwargs):

        res = {}

        for i, b in enumerate(beats):

            _b = np.array(b)
            if len(_b) < 25:
                _b = np.pad(_b, (0, 25-len(_b)), mode="edge")
            _1b = self.moving_average(hbutils.derivative(np.array(b), 1), 25)
            _2b = hbutils.derivative(np.array(_b), 2)

            assert len(_1b) == len(_2b) == len(_b), "%d, %d, %d" % (len(_b), len(_1b), len(_2b))

            _b -= _b.min()

            _b /= _b.max()

            # first point of interest is maximum in 1st dev
            p1 = np.argmax(_1b)
            # if there is a minimum that is not in index 0, cut it
            _1b_before_p1 = _1b[:p1]
            mins_before_p1 = argrelmin(_1b_before_p1)[0]
            if len(mins_before_p1) > 0:
                _b = _b[mins_before_p1[-1]:]
                _1b = _1b[mins_before_p1[-1]:]
                _2b = _2b[mins_before_p1[-1]:]
                p1 = p1 - mins_before_p1[-1]

            # define sys peak as first maximum after p1 in _b
            p2 = argrelmax(_b[p1:])[0][0] + p1
            # or the first minimum in _1b , whichever comes first
            p2 = min(p2, self.first_minmax_if_exists(_1b[p1:], "min")+p1)

            # first min after p2 in _1b
            p3 = self.first_minmax_if_exists(_1b[p2:], "min") + p2
            # first max after p3 in _1b
            p4 = p3
            p5 = p4
            maxes_after_p3 = argrelmax(_1b[p3:])[0]
            if len(maxes_after_p3)>0:
                p4 = maxes_after_p3[0] + p3
                mins_after_p4 = argrelmin(_1b[p4:])[0]
                if len(mins_after_p4) > 0:
                    p5 = mins_after_p4[0] + p4

            # adding sanity checks
            # defaulting to moot indexes if the function fails to locate fiducial points
            p1 = max(0, min(int(p1), len(_b) - 1))
            p2 = max(0, min(int(p2), len(_b) - 1))
            p3 = max(0, min(int(p3), len(_b) - 1))
            p4 = max(0, min(int(p4), len(_b) - 1))
            p5 = max(0, min(int(p1), len(_b) - 1))

            res[i] = {}
            res[i]["_b"] = _b.tolist()
            res[i]["_1b"] = _1b.tolist()
            res[i]["_2b"] = _2b.tolist()
            res[i]["p1"] = p1
            res[i]["systolic_peak_i"] = p2
            res[i]["systolic_peak_c"] = 1.0
            res[i]["dychrotic_notch_i"] = p3
            res[i]["dychrotic_notch_c"] = 1.0
            res[i]["diastolic_peak_i"] = p4
            res[i]["diastolic_peak_c"] = 1.0
            res[i]["p5"] = p5

        return res

    def hb_fiducial_point_detection(self, beats, **kwargs):

        res = {}

        for i, b in enumerate(beats):
            _b = np.array(b)
            _1b = hbutils.derivative(np.array(b), 1)
            _2b = hbutils.derivative(np.array(b), 2)
            
            _b -= _b.min()

            _b /= _b.max()

            res[i] = {}

            systolic_peak_i, systolic_peak_c = self.find_systolic_peak(_b, _1b, _2b)

            dychrotic_notch_i, dychrotic_notch_c = self.find_dychrotic_notch(_b, _1b, _2b, systolic_peak_i)

            diastolic_peak_i, diastolic_peak_c = self.find_dystolic_peak(_b, _1b, _2b, systolic_peak_i, dychrotic_notch_i)

            res[i]["_b"] = _b.tolist()
            res[i]["_1b"] = _1b.tolist()
            res[i]["_2b"] = _2b.tolist()
            res[i]["systolic_peak_i"] = int(systolic_peak_i)
            res[i]["systolic_peak_c"] = float(systolic_peak_c)
            res[i]["dychrotic_notch_i"] = int(dychrotic_notch_i)
            res[i]["dychrotic_notch_c"] = float(dychrotic_notch_c)
            res[i]["diastolic_peak_i"] = int(diastolic_peak_i)
            res[i]["diastolic_peak_c"] = float(diastolic_peak_c)

        return res

    def find_systolic_peak(self, signal, dev1, dev2, left_margin_p=0.05):
        """
        :param signal: signal for the beat, normalized so that it is in [0, 1]
        :param dev1: first derivative
        :param dev2: second derivative
        :return: index of peak
        """
        left_margin_offset = int(signal.shape[0] * left_margin_p)
        # get the first maximum after the biggest minimum in dev1
        big_min_in_dev1 = np.argwhere(dev1 == dev1.min())[0, 0]
        maxes_after_abs_min = argrelmax(dev1[big_min_in_dev1:]) + big_min_in_dev1
        found_confidence = 0.75

        # first_max_after_abs_min = 0
        first_max_after_abs_min = maxes_after_abs_min.flatten()[0]
        # if we find a zero-crossing in dev1 before the max, return that instead
        zero_cross_i = self.find_zero_crossing(dev1[left_margin_offset:first_max_after_abs_min], which="neg_to_pos")
        if zero_cross_i != 0:
            found_confidence = 1.0
            return left_margin_offset+zero_cross_i, found_confidence
        return first_max_after_abs_min, found_confidence

    def find_dychrotic_notch(self, signal, dev1, dev2, systolic_peak_i):
        # first attempt, if there is a clear point where the first derivative goes from positive to negative, take that
        signal_after_sp = signal[systolic_peak_i:]
        dev1_after_sp = dev1[systolic_peak_i:]
        dev2_after_sp = dev2[systolic_peak_i:]

        mins_dev1_after_sp = argrelmin(dev1_after_sp)[0]
        dev1_after_sp_before_first_min = np.array(dev1_after_sp)
        if mins_dev1_after_sp.shape[0] > 0:
            dev1_after_sp_before_first_min = dev1_after_sp[:mins_dev1_after_sp[0]]

        found_confidence = 0.0
        dychrotic_notch_index = self.find_zero_crossing(dev1_after_sp_before_first_min, which="pos_to_neg")
        if dychrotic_notch_index != 0:
            found_confidence = 1.0
        else:
            # then take the maximum in the 2nd derivative
            maxs = argrelmax(dev2_after_sp)[0]
            if maxs.shape[0] > 0:
                dychrotic_notch_index = maxs[0]
                found_confidence = 0.5

        if dychrotic_notch_index + systolic_peak_i >= len(signal)-2:
            found_confidence = 0.0

        return dychrotic_notch_index + systolic_peak_i, found_confidence

    def find_zero_crossing(self, s, which="pos_to_neg"):
        mask = (True, False) if which == "pos_to_neg" else (False, True)
        positive_mask = s > 0
        for i, _ in enumerate(positive_mask[:-1]):
            if positive_mask[i] == mask[0] and positive_mask[i + 1] == mask[1]:
                return i+1
        return 0

    def find_dystolic_peak(self, signal, dev1, dev2, systolic_peak_i, dychrotic_notch_i):
        dev1_after_dn = dev1[dychrotic_notch_i:]
        dev2_after_dn = dev2[dychrotic_notch_i:]

        maxs_dev1_after_sp = argrelmax(dev1_after_dn)[0]
        dev1_after_dn_before_first_max = np.array(dev1_after_dn)

        if maxs_dev1_after_sp.shape[0] > 0:
            dev1_after_dn_before_first_max = dev1_after_dn[:maxs_dev1_after_sp[0]]

        found_confidence = 0.0
        diastolic_peak_index = self.find_zero_crossing(dev1_after_dn_before_first_max, which="neg_to_pos")

        if diastolic_peak_index != 0:
            found_confidence = 1.0
        else:
            # then look at the zero crossing pos_to_neg in dev2
            diastolic_peak_index = self.find_zero_crossing(dev2_after_dn, which="pos_to_neg")
            if diastolic_peak_index != 0:
                found_confidence = 0.75
            else:
                # the first minimum in dev2 after the dychrotic_notch
                mins = argrelmin(dev2_after_dn)[0]
                if mins.shape[0] > 0:
                    diastolic_peak_index = mins[0]
                    found_confidence = 0.5

        if diastolic_peak_index + dychrotic_notch_i >= len(signal)-2:
            found_confidence = 0.0

        return diastolic_peak_index + dychrotic_notch_i, found_confidence


def plot_things(result, filename):

    dict_keys = list(result.keys())
    fig, ax = plt.subplots(nrows=3, figsize=(20, 4*3))
    offset_on_x = 0
    label1, label2, label3, label_sys, label_dynotch, label_dysys = "beat", "1st", "2nd", "systolic_peak", "dychrotic_notch", "diastolic_peak"
    label_p1, label_p5 = "max_gradient", "min_gradient"
    max_1 = 0.000000001
    max_2 = 0.000000001

    for i in range(len(dict_keys)):
        _b = np.array(result[i]["_b"])
        _1b = np.array(result[i]["_1b"])
        _2b = np.array(result[i]["_2b"])

        x = np.arange(0, len(_b), dtype=int) + offset_on_x

        p1 = 0
        p5 = 0
        if "p1" in result[i]:
            p1 = result[i]["p1"]
        systolic_peak_i = result[i]["systolic_peak_i"]
        dychrotic_notch_i = result[i]["dychrotic_notch_i"]
        dystolic_peak_i = result[i]["diastolic_peak_i"]
        if "p5" in result[i]:
            p5 = result[i]["p5"]

        ax[0].scatter([offset_on_x + p5], [_b[p5]], marker=">", color="pink", label=label_p5)
        ax[0].scatter([offset_on_x + p1], [_b[p1]], marker="s", color="orange", label=label_p1)
        ax[0].scatter([offset_on_x + systolic_peak_i], [_b[systolic_peak_i]], marker="x", color="red", label=label_sys)
        ax[0].scatter([offset_on_x + dychrotic_notch_i], [_b[dychrotic_notch_i]], marker="o", color="green",
                      label=label_dynotch)
        ax[0].scatter([offset_on_x + dystolic_peak_i], [_b[dystolic_peak_i]], marker="^", color="black",
                      label=label_dysys)
        ax[0].plot(x, _b, label=label1, c="b")
        ax[0].plot([offset_on_x, offset_on_x], [-0, 1], c="k", linestyle="dashed")

        ax1_range = (-0.1, 0.1)
        ax[1].plot(x, _1b, label=label2, c="b")
        ax[1].plot([offset_on_x, offset_on_x], ax1_range, c="k", linestyle="dashed")
        rect = patches.Rectangle((offset_on_x, ax1_range[0]), systolic_peak_i, ax1_range[1] - ax1_range[0], linewidth=1,
                                 edgecolor='r', facecolor='red', alpha=.25)
        ax[1].add_patch(rect)
        ax[1].scatter([offset_on_x + p5], [_1b[p5]], marker=">", color="pink", label=label_p5)
        ax[1].scatter([offset_on_x + p1], [_1b[p1]], marker="s", color="orange", label=label_p1)
        ax[1].scatter([offset_on_x + systolic_peak_i], [_1b[systolic_peak_i]], marker="x", color="red", label=label_sys)
        ax[1].scatter([offset_on_x + dychrotic_notch_i], [_1b[dychrotic_notch_i]], marker="o", color="green",
                      label=label_dynotch)
        ax[1].scatter([offset_on_x + dystolic_peak_i], [_1b[dystolic_peak_i]], marker="^", color="black",
                      label=label_dysys)

        ax[1].plot([offset_on_x, offset_on_x + _1b.shape[0]], [0, 0], c="k", linestyle="dashed")

        ax2_range = (-0.1, 0.1)
        rect = patches.Rectangle((offset_on_x, ax2_range[0]), systolic_peak_i, ax2_range[1] - ax2_range[0], linewidth=1,
                                 edgecolor='r', facecolor='red', alpha=.25)
        ax[2].add_patch(rect)
        ax[2].scatter([offset_on_x + p5], [_2b[p5]], marker=">", color="pink", label=label_p5)
        ax[2].scatter([offset_on_x + p1], [_2b[p1]], marker="s", color="orange", label=label_p1)
        ax[2].scatter([offset_on_x + systolic_peak_i], [_2b[systolic_peak_i]], marker="x", color="red", label=label_sys)
        ax[2].scatter([offset_on_x + dychrotic_notch_i], [_2b[dychrotic_notch_i]], marker="o", color="green",
                      label=label_dynotch)
        ax[2].scatter([offset_on_x + dystolic_peak_i], [_2b[dystolic_peak_i]], marker="^", color="black",
                      label=label_dysys)
        ax[2].plot(x, _2b, label=label3, c="b")
        ax[2].plot([offset_on_x, offset_on_x], ax2_range, c="k", linestyle="dashed")
        ax[2].plot([offset_on_x, offset_on_x + _2b.shape[0]], [0, 0], c="k", linestyle="dashed")

        offset_on_x += len(_b)

        if np.max(np.abs([_1b.min(), _1b.max()])) > max_1:
            max_1 = np.max(np.abs([_1b.min(), _1b.max()]))
        if np.max(np.abs([_2b.min(), _2b.max()])) > max_2:
            max_2 = np.max(np.abs([_2b.min(), _2b.max()]))

        if i == 0:
            label1, label2, label3, label_sys, label_dynotch, label_dysys, label_p1, label_p5 = "", "", "", "", "", "", "", ""

    ax[0].legend()
    ax[0].grid(linestyle="dashed")
    ax[0].set_xticks([])

    ax[1].legend()
    ax[1].grid(linestyle="dashed")
    ax[1].set_xticks([])
    ax[1].set_ylim((-max_1, max_1))

    ax[2].legend()
    ax[2].grid(linestyle="dashed")
    ax[2].set_xticks([])
    ax[2].set_ylim((-max_2, max_2))


    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    # plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey it's me")
    parser.add_argument("-d", "--beats_directory", help="Directory with extracted beats", default="/home/data/beats/")
    parser.add_argument("-o", "--output_folder", help="Directory where to store output", default="/home/data/fiducial_points/")
    parser.add_argument("-s", "--display", action='store_true', default=False)
    parser.add_argument("-r", "--force_redo", action='store_true', default=False)
    parser.add_argument("-p", "--params", action='store', default="params.yaml")
    args = parser.parse_args()

    params = yaml.load(open(args.params, "r"), Loader=yaml.FullLoader)

    fpd = SignalFiducialPointsDetector(sample_rate=params["frame_rate"])

    if args.force_redo:
        shutil.rmtree(args.output_folder, ignore_errors=True)
        os.makedirs(args.output_folder, exist_ok=True)

    users = os.listdir(args.beats_directory)
    users = list(filter(lambda x: os.path.isdir(os.path.join(args.beats_directory, x)), users))

    for i, user in enumerate(sorted(users)):
        # get all files
        user_fold = os.path.join(args.beats_directory, user)
        user_files = os.listdir(user_fold)
        user_files = filter(lambda x: os.path.isfile(os.path.join(user_fold, x)), user_files)

        # only retain allowed formats
        user_files = filter(lambda x: x.split(".")[-1] == "json", user_files)
        os.makedirs(os.path.join(args.output_folder, user), exist_ok=True)
        for file in sorted(user_files):
            filepath = os.path.join(user_fold, file)
            fname = file.split(".")[0]

            img_out_fpath = os.path.join(args.output_folder, user, fname + ".pdf")
            json_out_fpath = os.path.join(args.output_folder, user, fname + ".json")
            if not os.path.isfile(json_out_fpath) or not os.path.isfile(img_out_fpath):
                extracted_beats = json.load(open(filepath, "r"))
                sys.stdout.write("{} ({}/{}),{}\n".format(user, i + 1, len(users), file))
                sys.stdout.flush()

                keys = sorted(map(int, extracted_beats["hb_argrelmin"].keys()))
                list_of_beats = [extracted_beats["hb_argrelmin"][str(k)] for k in keys]

                r = fpd.hb_fp_detection_20191217(list_of_beats)
                json.dump(r, open(json_out_fpath, "w"))

                plot_things(r, filename=img_out_fpath)

            else:
                sys.stdout.write("Skipping, file %s exists already\n" % fname)
                sys.stdout.flush()
