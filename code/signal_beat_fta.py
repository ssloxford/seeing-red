import argparse
import json
import shutil

import matplotlib
import yaml

matplotlib.use("agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import os
import sys
from tslearn.metrics import dtw
from hb_utils import interpolate_beat


class SignalBeatQuality():

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def dwt_filter_with_reference(self, beats_dict, ref, threshold, interp_dim):
        good_beats = []
        mask = {}
        dwt_dist = {}

        keys = list(sorted(beats_dict.keys()))
        for k in keys:
            b = beats_dict[k]
            _b = np.array(b)
            _b = _b-_b.min()
            _b = _b/_b.max()

            x, _b = interpolate_beat(_b, interp_dim)

            dist = dtw(_b, ref)
            dwt_dist[k] = "%.2f" % dist

            if dist < threshold:
                good_beats.append(b)
                mask[k] = True
            else:
                mask[k] = False

        return good_beats, mask, dwt_dist

    def fid_points_location_filter(self, beats_dict, fid_points_dict, tol=10):
        good_beats = []
        mask = {}
        fpf = {}

        keys = list(sorted(beats_dict.keys()))
        for k in keys:
            b = beats_dict[k]
            fp = fid_points_dict[k]
            if fp["systolic_peak_i"] >= len(b)-tol:
                mask[k] = False
                fpf[k] = "sp%d" % fp["systolic_peak_i"]
            elif fp["dychrotic_notch_i"] >= len(b) - tol:
                mask[k] = False
                fpf[k] = "dn%d" % fp["dychrotic_notch_i"]
            elif fp["diastolic_peak_i"] >= len(b) - tol:
                mask[k] = False
                fpf[k] = "dp%d" % fp["diastolic_peak_i"]
            else:
                mask[k] = True
                good_beats.append(b)
                fpf[k] = "ok"
        return good_beats, mask, fpf

    def less_than_three_peaks(self, beats_dict):
        good_beats = []
        mask = {}
        descr = {}

        keys = list(sorted(beats_dict.keys()))
        for k in keys:
            b = np.array(beats_dict[k])
            maxes = argrelmax(b)
            if len(maxes[0]) >= 3:
                mask[k] = False
                descr[k] = "%d mx" % len(maxes[0])
            else:
                mask[k] = True
                descr[k] = "%d mx" % len(maxes[0])
                good_beats.append(b)

        return good_beats, mask, descr

    def systolic_peak_is_not_max_filter(self, beats_dict, fid_points_dict):
        good_beats = []
        mask = {}
        descr = {}

        keys = list(sorted(beats_dict.keys()))
        for k in keys:
            b = np.array(beats_dict[k])

            fp = fid_points_dict[k]

            if b[fp["systolic_peak_i"]] != b.max():
                mask[k] = False
                descr[k] = "no"
            else:
                good_beats.append(b.tolist())
                mask[k] = True
                descr[k] = "ok"

        return good_beats, mask, descr

    def filter_bpm_min(self, beats_dict, max_beat_length):
        good_beats = []
        distances = {}
        mask = {}

        keys = list(sorted(beats_dict.keys()))
        for k in keys:
            distances[k] = "%d" % len(beats_dict[k])
            if len(beats_dict[k]) <= max_beat_length:
                good_beats.append(beats_dict[k])
                mask[k] = True
            else:
                mask[k] = False

        return good_beats, mask, distances

    def filter_uneven_ends(self, beats, threshold=.25):
        good_beats = []
        distances = []
        mask = []
        for b in beats:
            _b = np.array(b)
            _b = _b - _b.min()
            _b = _b / _b.max()
            # x, _b = interpolate_beat(_b, 100)
            first_last_gap = np.abs(_b[0] - _b[-1])
            if first_last_gap <= threshold:
                good_beats.append(b)
                mask.append(True)
            else:
                mask.append(False)
            distances.append(first_last_gap)
        return good_beats, mask, np.array(distances)


def plot_things1(lines, scatters, filename):
    fig, ax = plt.subplots(nrows=len(lines), figsize=(12, 12))
    for i in range(len(lines)):
        # ith subplot
        lines_tp = lines[i]
        for j, _ in enumerate(lines_tp):
            # jth line
            x, y, label, color = lines_tp[j]
            ax[i].plot(x, y, label=label, c=color)

    for i in range(len(scatters)):
        # ith subplot
        scatters_tp = scatters[i]
        for j, _ in enumerate(scatters_tp):
            # jth line
            x, y, label, marker = scatters_tp[j]
            ax[i].scatter(x, y, label=label, marker=marker)

    for i in range(max(len(scatters), len(lines))):
        ax[i].legend(loc="upper right")
        ax[i].grid()

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    # plt.clf()
    plt.close(fig)


def plot_things(beats_original, beats_ftaed, reference, headers, distances, masks, filename):
    x_counter = 0
    plt.figure(figsize=(30, 5))
    _, i_ref = interpolate_beat(reference, 30)

    text_gap = 0.05

    for i, h in enumerate(headers):
        plt.text(-200-len(i_ref), -(i+1)*text_gap, h)

    keys_org = list(sorted(map(int, beats_original.keys())))
    keys_fta = list(sorted(map(int, beats_ftaed.keys())))

    for i, key in enumerate(keys_org):
        b = beats_original[str(key)]
        b = np.array(b)
        b = b-b.min()
        b = b/b.max()
        color = "b" if np.all(masks[i, :]) else "r"
        plt.plot(np.arange(x_counter, x_counter+len(b)), b, c=color)

        for j in range(masks.shape[1]):
            plt.text(x_counter, -(j+1)*text_gap, distances[i][j], c="black" if masks[i, j] else "red")
        x_counter += len(b)

    plt.ylim(-.3, 1.05)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey it's me")
    parser.add_argument("-f", "--filename", help="Feed me if you want to visualize single file")
    parser.add_argument("-d", "--beats_directory", help="Directory with subfolders for identity", default="/home/data/beats/")
    parser.add_argument("-o", "--output_folder", help="Directory where to store filtered csvs", default="/home/data/beats-post-FTA/")
    parser.add_argument("-r", "--force_redo", action='store_true', default=False)
    parser.add_argument("-p", "--params", action='store', default="params.yaml")

    args = parser.parse_args()

    params = yaml.load(open(args.params, "r"), Loader=yaml.FullLoader)

    bq = SignalBeatQuality(sample_rate=params["frame_rate"])

    reference = pd.read_csv("/home/data/beat_visuals/NOFTA.csv", index_col=False).values[..., :-1].mean(axis=0)
    interpolate_dim = reference.shape[0]

    threshold_dwt = 2.0
    threshold_ends = 0.25
    min_bpm = 30
    max_beat_length = bq.sample_rate / (min_bpm / 60)  # should be 90 for bpm of 40

    if args.force_redo:
        shutil.rmtree(args.output_folder, ignore_errors=True)
        os.makedirs(args.output_folder, exist_ok=True)

    if args.filename is None:
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

                img_fpath = os.path.join(args.output_folder, user, fname + ".png")
                json_out_fpath = os.path.join(args.output_folder, user, fname + ".json")
                if not os.path.isfile(json_out_fpath) or not os.path.isfile(img_fpath):
                    extracted_beats = json.load(open(filepath, "r"))
                    sys.stdout.write("{} ({}/{}),{}\n".format(user, i + 1, len(users), file))
                    sys.stdout.flush()
                    img_fpath = os.path.join(args.output_folder, user, file.split(".")[0] + ".png")

                    # load fiducial points
                    fiducial_points = json.load(open("/home/data/fiducial_points/%s/%s" % (user, file), "r"))

                    # first filter is based on max bpm set to 100
                    _, m_bpm, lengths = bq.filter_bpm_min(extracted_beats["hb_argrelmin"], max_beat_length=max_beat_length)
                    _, m_dwt, dwt_dist = bq.dwt_filter_with_reference(extracted_beats["hb_argrelmin"], ref=reference, threshold=threshold_dwt, interp_dim=interpolate_dim)
                    _, m_fp, fpfd_descr = bq.fid_points_location_filter(extracted_beats["hb_argrelmin"], fid_points_dict=fiducial_points)
                    # _, m_spm, spm_descr = bq.systolic_peak_is_not_max_filter(extracted_beats["hb_argrelmin"], fid_points_dict=fiducial_points)
                    _, m_3rdp, thirdp_descr = bq.less_than_three_peaks(extracted_beats["hb_argrelmin"])
                    # _, m_ends_gaps, ends_gap_dist = bq.filter_uneven_ends(list_of_beats, threshold=threshold_ends)

                    keys = list(sorted(map(int, extracted_beats["hb_argrelmin"].keys())))
                    all_masks = [[
                        m_bpm[j],
                        m_fp[j],
                        # m_spm[j],
                        m_dwt[j],
                        m_3rdp[j]
                    ] for j in map(str, keys)]
                    all_masks = np.array(all_masks)
                    all_descrs = [[
                        lengths[j],
                        fpfd_descr[j],
                        # spm_descr[j],
                        dwt_dist[j],
                        dwt_dist[j],
                        thirdp_descr[j]
                    ] for j in map(str, keys)]

                    headers = ["bpm", "fpf", "dwt", "3maxp"]

                    ftaed_beats = dict()
                    ftaed_beats["hb_argrelmin"] = dict()
                    for j, k in enumerate(keys):
                        if np.all(all_masks[j, :]):
                            ftaed_beats["hb_argrelmin"][str(k)] = extracted_beats["hb_argrelmin"][str(k)]

                    plot_things(
                        extracted_beats["hb_argrelmin"],
                        ftaed_beats["hb_argrelmin"],
                        reference,
                        headers=headers,
                        distances=all_descrs,
                        masks=all_masks,
                        filename=img_fpath)
                    json.dump(ftaed_beats, open(json_out_fpath, "w"))

                    # plot_things(ltp[0], stp[0], filename=img_fpath)
                    #df = pd.DataFrame(feat, columns=f_names)
                    #df.to_csv(json_out_fpath, sep=",", float_format="%.4f", index=False)

                else:
                    sys.stdout.write("Skipping, file %s exists already\n" % fname)
                    sys.stdout.flush()


