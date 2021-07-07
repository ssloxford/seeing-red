import argparse
import os
import shutil

import cv2
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import FastICA, PCA
from joblib import Parallel, delayed, cpu_count
import yaml


class SignalExtractor():

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        pass

    def red_channel_mean(self, frames, **kwargs):
        signal = []
        for frame_bgr in frames:
            mean_of_r_ch = frame_bgr[..., 2].mean()
            signal.append(mean_of_r_ch)
        signal = np.array(signal)
        samples_to_skip = kwargs["initial_skip_seconds"]*self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def green_channel_mean(self, frames, **kwargs):
        signal = []
        for frame_bgr in frames:
            mean_of_r_ch = frame_bgr[..., 1].mean()
            signal.append(mean_of_r_ch)
        signal = np.array(signal)
        samples_to_skip = kwargs["initial_skip_seconds"]*self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def green_channel_mean_upper_half(self, frames, **kwargs):
        signal = []
        for frame_bgr in frames:
            mean_of_r_ch = frame_bgr[:frame_bgr.shape[0]//2, : , 1].mean()
            signal.append(mean_of_r_ch)
        signal = np.array(signal)
        samples_to_skip = kwargs["initial_skip_seconds"]*self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def luma_component_mean(self, frames, **kwargs):
        signal = []
        for frame_bgr in frames:
            img_ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            mean_of_luma = img_ycrcb[..., 0].mean()
            signal.append(mean_of_luma)

        signal = np.array(signal)
        samples_to_skip = kwargs["initial_skip_seconds"] * self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def ica_decomposition(self, frames, **kwargs):
        s_r, s_g, s_b = [], [], []
        for frame_bgr in frames:
            b, g, r = frame_bgr.mean(axis=0).mean(axis=0)
            s_r.append(r)
            s_b.append(b)
            s_g.append(g)

        s_r = np.array(s_r).reshape(1, -1)
        s_b = np.array(s_b).reshape(1, -1)
        s_g = np.array(s_g).reshape(1, -1)

        fica = FastICA(n_components=1)
        stackd = np.concatenate((s_r, s_b, s_g), axis=0).T
        signal = fica.fit_transform(stackd).flatten()
        samples_to_skip = kwargs["initial_skip_seconds"] * self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal

    def red_ch_threshold(self, frames, n_calib_frames=90, perc=80, **kwargs):
        # Average the per-frame <perc> percentile of the red channel over the first <calib> frames
        calib_vals = []
        calib_count = 0
        # jesus don't judge me for this
        while calib_count <= n_calib_frames:
            b, g, r = cv2.split(frames[calib_count])
            r = r.flatten()
            cval = np.percentile(r, perc)
            calib_vals.append(cval)
            calib_count += 1

        threshold = np.mean(calib_vals)
        signal = []
        img_h, img_w, _ = frames[0].shape
        for frame in frames:
            b, g, r = cv2.split(frame)
            mask_gt_threshold = r>threshold
            signal.append(mask_gt_threshold.astype(int).sum()/(img_h*img_w))

        signal = np.array(signal)
        signal = signal[kwargs["initial_skip_seconds"]*self.sample_rate:]  # ignore first second because of auto exposure
        return signal

    def small_boxes_man(self, frames, **kwargs):
        n_boxes = kwargs["n_boxes"]

        frame_w = frames[0].shape[1]
        frame_h = frames[0].shape[0]

        assert frame_w / n_boxes == frame_w // n_boxes, frame_h / n_boxes == frame_h // n_boxes

        box_h, box_w = frame_h // n_boxes, frame_w // n_boxes

        signal = np.zeros((len(frames), n_boxes, n_boxes))
        for i, frame_bgr in enumerate(frames):
            img_ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            for j in range(n_boxes):
                for k in range(n_boxes):
                    cell = img_ycrcb[j * box_h:(j + 1) * box_h, k * box_w:(k + 1) * box_w, :]
                    cell = cell[..., 0].mean()
                    signal[i, j, k] = cell

        signal = signal.reshape(signal.shape[0], -1)

        pca = PCA(n_components=1)
        signal = pca.fit_transform(signal).flatten()

        samples_to_skip = kwargs["initial_skip_seconds"] * self.sample_rate
        signal = signal[samples_to_skip:]  # ignore first second because of auto exposure
        return signal


def visualize_signal(signals, labels, output_fname, title=""):
    fig, ax = plt.subplots(nrows=len(signals), figsize=(16, 4*len(signals)))

    for i, signal in enumerate(signals):
        # for visualization normalize signal
        to_plot = (signal-signal.mean())/ signal.std()

        ax[i].plot(range(signal.shape[0]), to_plot, label=labels[i])
        ax[i].legend()
        ax[i].grid(linestyle='dashed',)

    plt.savefig(output_fname, bbox_inches="tight")
    plt.close(fig)
    return True


# lets make this one parallel
def process_single_file(file, user_fold, output_folder, params, users, se):
    filepath = os.path.join(user_fold, file)
    fname = file.split(".")[0]
    csv_fpath = os.path.join(output_folder, users, fname + ".csv")

    if not os.path.isfile(csv_fpath):
        columns, extracted_s = [], []

        list_of_frames = []
        vidcap = cv2.VideoCapture(filepath)
        success, frame = vidcap.read()
        h, w, _ = frame.shape
        ioo = 0
        while success:
            list_of_frames.append(frame)
            success, frame = vidcap.read()
            ioo+=1

        vidcap.release()
        for i,extractor in enumerate(params["extractor"]):
            columns.append(extractor["name"])
            sys.stdout.write(
                "{} ({}/{}),{},{}\n".format(users, i + 1, len(users), file, extractor["name"]))
            sys.stdout.flush()
            assert len(extractor["functions"]) == 1, "Only one extractor function is supported, check config.json"
            for fun_name in extractor["functions"]:
                fun = getattr(se, fun_name)
                f_output = fun(frames=list_of_frames, **extractor["parameters"])
            extracted_s.append(f_output.tolist())
        extracted_s = np.array(extracted_s) * -1
        assert extracted_s.ndim == 2, "Different functions resulted in different length of extracted signal"
        df = pd.DataFrame(extracted_s.T, columns=columns)
        df.to_csv(csv_fpath, sep=",", float_format="%.4f", index=False)

        # visualize_signal(
        #    extracted_s,
        #    labels=columns,
        #    output_fname=img_fpath, )
    else:
        sys.stdout.write("Skipping, file %s exists already\n" % fname)
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey it's me")
    parser.add_argument("-f", "--filename", help="Feed me if you want to visualize single file")
    parser.add_argument("-d", "--video_directory", help="Directory with subfolders for identity", default="/home/data/videos/")
    parser.add_argument("-o", "--output_folder", help="Directory where to store computed csvs", default="/home/data/extracted/")
    parser.add_argument("-s", "--display", action='store_true', default=False)
    parser.add_argument("-r", "--force_redo", action='store_true', default=False)
    parser.add_argument("-n_cpu", "--number_of_cpus", action='store', type=int, default=2)
    parser.add_argument("-p", "--params", action='store', default="params.yaml")
    args = parser.parse_args()

    if args.number_of_cpus > cpu_count():
        args.number_of_cpus = cpu_count()

    params = yaml.load(open(args.params, "r"), Loader=yaml.FullLoader)

    allowed_video_fmts = ["mov", "mp4"]
    print("Allowed video formats", allowed_video_fmts)
    allowed_video_fmts.extend(list(map(lambda x: x.upper(), allowed_video_fmts)))

    se = SignalExtractor(sample_rate=params["frame_rate"])

    if args.force_redo:
        shutil.rmtree(args.output_folder, ignore_errors=True)
        os.makedirs(args.output_folder, exist_ok=True)

    if args.filename is None:
        users = os.listdir(args.video_directory)
        users = list(filter(lambda x: os.path.isdir(os.path.join(args.video_directory, x)), users))
        for i, user in enumerate(sorted(users)):
            # get all files
            user_fold = os.path.join(args.video_directory, user)
            user_files = os.listdir(user_fold)
            user_files = filter(lambda x: os.path.isfile(os.path.join(user_fold, x)), user_files)

            # only retain allowed formats
            user_files = list(filter(lambda x: x.split(".")[-1] in allowed_video_fmts, user_files))
            os.makedirs(os.path.join(args.output_folder, user), exist_ok=True)

            Parallel(n_jobs=args.number_of_cpus)(delayed(process_single_file)(
                file, user_fold, args.output_folder, params, users, SignalExtractor(sample_rate=params["frame_rate"]))
                               for file in sorted(user_files))

            for file in sorted(user_files):
                n_extractors = len(params["extractor"])
                csv_fname = file.split(".")[0] + ".csv"
                pdf_fname = file.split(".")[0] + ".pdf"
                df = pd.read_csv(os.path.join(args.output_folder, user, csv_fname), index_col=False)
                df.iloc[30:].plot(kind="line", subplots=True, figsize=(16, 4*n_extractors), layout=(n_extractors, 1), grid=True)
                plt.savefig(os.path.join(args.output_folder, user, pdf_fname), bbox_inches="tight")
                plt.close()

    if args.filename is not None:

        list_of_frames = []
        vidcap = cv2.VideoCapture(args.filename)
        success, frame = vidcap.read()
        ioo = 0
        while success:
            list_of_frames.append(frame)
            success, frame = vidcap.read()
            ioo += 1

        # then only do filename
        print("rcm")
        signal1 = se.red_channel_mean(frames=list_of_frames)*-1

        print("ica")
        signal3 = se.ica_decomposition(frames=list_of_frames)*-1

        print("luma")
        signal4 = se.luma_component_mean(frames=list_of_frames)*-1

        visualize_signal(
            [signal1, signal3, signal4],
            labels=["red_ch_mean", "ica", "luma"],
            output_fname="here.pdf", title="Extracted ppgs")
        #get_ppg_only_r(frames)
        #get_ppg(args.filename, frames)







