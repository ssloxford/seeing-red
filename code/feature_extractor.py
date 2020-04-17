import argparse
import json

import matplotlib
import yaml

matplotlib.use("agg")
import pandas as pd
import numpy as np
import os
import sys
from hb_utils import interpolate_beat
import hb_utils as hbutils

# References:
# [1] Reşit Kavsaoǧlu, A., Polat, K.,
# & Recep Bozkurt, M. (2014). A novel feature ranking algorithm for
# biometric recognition with PPG signals.

class FeatureExtractor():

    def __init__(self, ):
        pass

    def integrate(self, y_vals, h=1):
        i = 1
        total = y_vals[0] + y_vals[-1]
        for y in y_vals[1:-1]:
            if i % 2 == 0:
                total += 2 * y
            else:
                total += 4 * y
            i += 1
        return total * (h / 3.0)

    def slope(self, x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        return m

    def derivative(self, signal, index=1):
        # Implemented according to [1]
        if index == 1:
            return [a - b for a, b in zip(signal, signal[1:])] + [0]
        elif index == 2:
            return [0] + [c + a - 2 * b for a, b, c in zip(signal, signal[1:], signal[2:])] + [0]
        else:
            raise ValueError("Only support first or second derivatives")

    def f_cheating(self, beats, beat_det_names, **kwargs):

        feature_names = [
            "meta_counter",
            "max",
            "min",
            "max-min",
            "length"
        ]
        features = []

        for beat_det_name in beat_det_names:
            for beat_counter in beats[beat_det_name].keys():
                this_b_features = [beat_counter]
                signal = np.array(beats[beat_det_name][beat_counter])
                this_b_features.extend([
                    signal.max(),
                    signal.min(),
                    signal.max()-signal.min(),
                    signal.shape[0]
                ])
                features.append(this_b_features)

        features = np.array(features)
        if features.shape[0]==0:
            features = np.empty((0, len(feature_names)))

        assert features.shape[1] == len(feature_names), "%s, %d" % (features.shape, len(feature_names))
        return features, feature_names

    def f_widths(self, beats, beat_det_names, **kwargs):
        widths_at = kwargs["widths_at"]
        interp_dim = kwargs["interp_dim"]

        feature_names = ["meta_counter"]
        widths_at = list(map(int, widths_at))
        feature_names.extend(["w@%d" % x for x in widths_at])

        features = []

        for beat_det_name in beat_det_names:

            for beat_counter in beats[beat_det_name].keys():
                this_b_features = [beat_counter]
                signal = np.array(beats[beat_det_name][beat_counter])

                signal -= signal.min()
                signal /= signal.max()

                x, y = interpolate_beat(signal, interp_dim)
                for x in widths_at:
                    index_x = np.argwhere(y >= float(x)/100.0).flatten()
                    this_b_features.append(np.abs(index_x[0] - index_x[-1]))

                features.append(this_b_features)

        features = np.array(features)
        if features.shape[0]==0:
            features = np.empty((0, len(feature_names)))

        assert features.shape[1] == len(feature_names), "%d, %d" % (features.shape[1], len(feature_names))
        return features, feature_names


    def f_fft(self, beats, beat_det_names, **kwargs):
        interp_dim = kwargs["interp_dim"]

        feature_names = ["meta_counter"]
        feature_names.extend(["bin_%d" % i for i in range(interp_dim)])

        features = []

        for beat_det_name in beat_det_names:

            for beat_counter in beats[beat_det_name].keys():
                this_b_features = [beat_counter]
                signal = np.array(beats[beat_det_name][beat_counter])

                signal -= signal.min()
                signal /= signal.max()

                x, y = interpolate_beat(signal, interp_dim)

                sp = np.fft.fft(y)
                fft_magnitudes = np.abs(sp)
                this_b_features.extend(fft_magnitudes)

                features.append(this_b_features)

        features = np.array(features)
        if features.shape[0]==0:
            features = np.empty((0, len(feature_names)))

        assert features.shape[1] == len(feature_names), "%d, %d" % (features.shape[1], len(feature_names))
        return features, feature_names

    def f_fiducial_points(self, beats, beat_det_names, **kwargs):
        interp_dim = kwargs["interp_dim"]
        fiducial_points = kwargs["fiducial_points"]

        feature_names = [
            "meta_counter",
            "peak_to_peak_t",  # From [1] as delta T
            "systolic_peak_index",  # From [1], as y
            "dychrotic_notch_index",   # From [1] as t1, time to first peak
            "diastolic_peak_index",   # From [1] as t3, diastolic peak index
            "A2_area",  # From [1], but instead of notch we do to diastolic peak
            "A1_area",  # From [1], but we do from diastolic peak down as opposed to notch down
            "A2_A1_ratio",
            "a1",   # Maximum of first derivative
            "b1",   # Minimum of first derivative
            "ta1",  # Index of a1
            "tb1",  # index of b1
            "a2",   # Maximum value of second derivative
            "b2",   # Minimum value of second derivative
            "ta2",  # Index of a2
            "tb2",  # Index of b2
            "b2_a2",     # b2 / a2
            "systolic_peak_c",
            "dychrotic_notch_c",
            "diastolic_peak_c"
        ]

        features = []

        for beat_det_name in beat_det_names:

            for beat_counter in beats[beat_det_name].keys():
                this_b_features = [beat_counter]
                signal = np.array(beats[beat_det_name][beat_counter])
                signal -= signal.min()
                signal /= signal.max()
                x, y = interpolate_beat(signal, interp_dim)

                scale_factor = interp_dim/signal.shape[0]

                systolic_peak_index = int(fiducial_points[beat_counter]["systolic_peak_i"]*scale_factor)
                systolic_peak_conf = fiducial_points[beat_counter]["systolic_peak_c"]
                systolic_peak_value = y[systolic_peak_index]

                dychrotic_notch_index = int(fiducial_points[beat_counter]["dychrotic_notch_i"]*scale_factor)
                dychrotic_notch_conf = fiducial_points[beat_counter]["dychrotic_notch_c"]
                dychrotic_notch_value = y[dychrotic_notch_index]

                diastolic_peak_index = int(fiducial_points[beat_counter]["diastolic_peak_i"]*scale_factor)
                diastolic_peak_conf = fiducial_points[beat_counter]["diastolic_peak_c"]
                diastolic_peak_value = y[diastolic_peak_index]

                peak_to_peak = np.abs(diastolic_peak_index - systolic_peak_index)

                a1_area = np.trapz(y[:dychrotic_notch_index])
                a2_area = np.trapz(y[dychrotic_notch_index:])
                area_ratio = a2_area/a1_area

                first_deriv = self.derivative(y, 1)
                second_deriv = self.derivative(y, 2)
                a1 = np.max(first_deriv)
                b1 = np.min(first_deriv)
                ta1 = np.argmax(first_deriv)
                tb1 = np.argmin(first_deriv)
                a2 = np.max(second_deriv)
                b2 = np.min(second_deriv)
                ta2 = np.argmax(second_deriv)
                tb2 = np.argmin(second_deriv)
                b2_a2 = b2 / a2

                this_b_features.extend([
                    peak_to_peak,
                    systolic_peak_index,
                    dychrotic_notch_index,
                    diastolic_peak_index,
                    a1_area,
                    a2_area,
                    area_ratio,
                    a1,
                    b1,
                    ta1,
                    tb1,
                    a2,
                    b2,
                    ta2,
                    tb2,
                    b2_a2
                ])

                this_b_features.extend([
                    systolic_peak_conf,
                    dychrotic_notch_conf,
                    diastolic_peak_conf
                ])

                features.append(this_b_features)

        features = np.array(features)
        if features.shape[0]==0:
            features = np.empty((0, len(feature_names)))
        assert features.shape[1] == len(feature_names), "%d, %d" % (features.shape[1], len(feature_names))
        return features, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey it's me")
    parser.add_argument("-o", "--output_folder", help="Directory where to store filtered csvs",
                        default="/home/data/features/")
    parser.add_argument("-s", "--display", action='store_true', default=False)
    parser.add_argument("-r", "--force_redo", action='store_true', default=False)
    parser.add_argument("-p", "--params", action='store', default="params.yaml")
    args = parser.parse_args()

    params = yaml.load(open(args.params, "r"), Loader=yaml.FullLoader)

    fe = FeatureExtractor()

    # let's do the extraction twice in order to save both
    # FTA-filtered samples and non-FTA-filtered ones

    peaks_folders = [
        "/home/data/beats/",
        "/home/data/beats-post-FTA/"
    ]

    ffile_postfixes = ["", "-FTA"]

    fiducial_points_folder = "/home/data/fiducial_points/"

    os.makedirs(args.output_folder, exist_ok=True)

    for pf, postfix in zip(peaks_folders, ffile_postfixes):

        hbutils.delete_all_subdirs(args.output_folder)

        users = os.listdir(pf)
        users = list(filter(lambda x: os.path.isdir(os.path.join(pf, x)), users))

        for i, user in enumerate(sorted(users)):
            # get all files
            user_fold = os.path.join(pf, user)
            user_files = os.listdir(user_fold)
            user_files = filter(lambda x: os.path.isfile(os.path.join(user_fold, x)), user_files)

            # only retain allowed formats
            user_files = filter(lambda x: x.split(".")[-1] == "json", user_files)

            for file in sorted(user_files):
                filepath = os.path.join(user_fold, file)
                fiducial_points_file = os.path.join(fiducial_points_folder, user, file)
                if not os.path.isfile(fiducial_points_file):
                    sys.stdout.write("Could not find fiducial points file for %s\n")
                    continue

                fname = file.split(".")[0]
                for fextractor in params["feature_extractor"]:
                    fun_name = fextractor["name"]
                    os.makedirs(os.path.join(args.output_folder, fun_name, user), exist_ok=True)
                    csv_out_fpath = os.path.join(args.output_folder, fun_name, user, fname + ".csv")
                    extracted_beats = json.load(open(filepath, "r"))
                    fiducial_points = json.load(open(fiducial_points_file, "r"))
                    fun = getattr(fe, fun_name)
                    features, fnames = fun(extracted_beats, ["hb_argrelmin"], fiducial_points=fiducial_points, **fextractor["params"])
                    df = pd.DataFrame(features, columns=fnames)
                    df.to_csv(csv_out_fpath, sep=",", float_format="%.6f", index=False)
                    sys.stdout.write("{} ({}/{}),{},{}\n".format(user, i + 1, len(users), file, fextractor["name"]))
                    sys.stdout.flush()

        # now merge into single file
        hbutils.merge_features_into_file(args.output_folder, postfix=postfix)

    # delete all temp subdirectories in /home/data/features
    hbutils.delete_all_subdirs(args.output_folder)
