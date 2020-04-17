import argparse
import shutil

import matplotlib

matplotlib.use("agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.signal import lfilter, filtfilt
import scipy.fftpack
import os
import sys
import yaml


class SignalPreprocessor():

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.shorter_names = {
            "hpf": "butter_highpass_filter",
            "lpf": "butter_lowpass_filter",
            "maf": "moving_average_flat",
            "diff_pad": "minus_with_pad",
            "fft": "fft",
            "roll_avg": "rolling_average",
            "sub": "subtract",
            "bandpass": "butter_bandpass_filter",
            "imf": "increase_main_freq",
            "cut_start": "cut_start",
            "bpf_bpm": "bandpass_bpm"
        }

    def bandpass_bpm(self, signal, multiplier, mincut, order, **kwargs):

        no_nan_signal = np.array(signal)
        n_nan = 0
        if np.any(np.isnan(signal)):
            n_nan = signal[np.isnan(signal)].shape[0]
            no_nan_signal = signal[~np.isnan(signal)]

        T = 1.0 / self.sample_rate
        N = no_nan_signal.shape[0]
        signal_fft = np.abs(scipy.fftpack.fft(no_nan_signal))[:N // 2]
        signal_fft = signal_fft / signal_fft.max()

        freq_x = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

        max_cutoff = freq_x[np.argmax(signal_fft)] * multiplier
        y = self.butter_highpass_filter(no_nan_signal, max_cutoff, order)
        y = np.concatenate((np.full(n_nan, np.nan), y), axis=0)

        return y

    def cut_start(self, signal, seconds, **kwargs):
        n_frames = self.sample_rate * seconds
        return np.concatenate((np.full(n_frames, np.nan), signal[n_frames:]), axis=0)

    def rolling_average(self, signal, **kwargs):
        window_size_seconds = kwargs["window_size_seconds"]
        window_size = int(window_size_seconds * self.sample_rate)
        if window_size % 2 == 0:
            window_size += 1
        y = np.convolve(signal, np.ones(window_size), 'valid') / window_size
        y = np.pad(y, [((window_size - 1) // 2, (window_size - 1) // 2)], mode='edge')
        return y

    def butter_bandpass_filter(self, signal, lowcut, highcut, order, **kwargs):
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, signal)
        return y

    def subtract(self, signal, **kwargs):
        original_signal = kwargs["prev_x"]
        assert signal.shape == original_signal.shape
        y = original_signal-signal
        return y

    def butter_highpass_filter(self, signal, cutoff, order, **kwargs):
        no_nan_signal = np.array(signal)
        n_nan = 0
        if np.any(np.isnan(signal)):
            n_nan = signal[np.isnan(signal)].shape[0]
            no_nan_signal = signal[~np.isnan(signal)]
        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, no_nan_signal)
        y = np.concatenate((np.full(n_nan, np.nan), y), axis=0)
        return y

    def butter_lowpass_filter(self, signal, low, filter_order, **kwargs):
        nyq = 0.5 * self.sample_rate
        normal_cutoff = low / nyq
        b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, signal)
        return y

    def minus_with_pad(self, x, pad, **kwargs):
        prev_x_padded = np.pad(kwargs["prev_x"], [(pad, 0)], mode='constant', constant_values=0)
        x_padded = np.pad(x, [(0, pad)], mode='constant', constant_values=0)
        y = prev_x_padded - x_padded
        y = y[pad:]
        return y

    def moving_average_flat(self, x, window_size=10, **kwargs):
        assert window_size % 2 == 1, "Odd number for window size in moving average pls"
        conv = np.convolve(x, np.ones(window_size), 'valid') / window_size
        return np.pad(conv, [((window_size-1)//2, (window_size-1)//2)], mode='constant', constant_values=0)

    def fft(self, signal, **kwargs):
        fourierTransform = np.fft.fft(signal) / len(signal)  # Normalize amplitude
        fourierTransform = fourierTransform[range(int(len(signal) / 2))]  # Exclude sampling frequency

        tpCount = len(signal)
        values = np.arange(int(tpCount / 2))
        timePeriod = tpCount / self.sample_rate
        frequencies = values / timePeriod

        return np.abs(fourierTransform[:200])

    def increase_main_freq(self, signal, **kwargs):
        pad = kwargs["pad"]
        mult = kwargs["mult"]
        fourierTransform = np.fft.fft(signal)  # Normalize amplitude
        # fourierTransform = fourierTransform[range(int(len(signal) / 2))]  # Exclude sampling frequency

        i_of_max = np.argmax(fourierTransform[10:50])+10

        fourierTransform[i_of_max-pad:i_of_max+pad] = fourierTransform[i_of_max-pad:i_of_max+pad] * mult
        y = np.fft.ifft(fourierTransform)
        return y


def visualize_signal(signals, labels, output_fname, title=""):
    plt.figure()
    plt.title(title)
    l_tp = [
        "luma_mean",
        "luma_mean>cut_start",
        "luma_mean>cut_start>hpf",
        "luma_mean>cut_start>hpf>bpf_bpm",
        "luma_mean>roll_avg>sub>lpf>cut_start"
    ]
    n_plots = len(l_tp)
    fig, ax = plt.subplots(nrows=n_plots, figsize=(6, int(n_plots*2)))
    plot_c = 0
    printed_already = []

    for i, signal in enumerate(signals):
        # for visualization normalize signal
        # print(labels[i], printed_already)
        if labels[i] in l_tp and labels[i] not in printed_already:

            # to_plot = (signal-signal.mean())/ signal.std()
            to_plot = signal
            ax[plot_c].plot(range(len(signal)), to_plot, label=labels[i])
            printed_already.append(labels[i])
            ax[plot_c].grid(linestyle='dashed',)
            ax[plot_c].legend(loc="lower right")
            ax[plot_c].set_xlim((-10, len(signal)+10))
            plot_c += 1

    plt.tight_layout()
    plt.savefig(output_fname, bbox_inches="tight")
    plt.close("all")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey it's me")
    parser.add_argument("-f", "--filename", help="Feed me if you want to visualize single file")
    parser.add_argument("-d", "--signals_directory", help="Directory with subfolders for identity", default="/home/data/extracted/")
    parser.add_argument("-o", "--output_folder", help="Directory where to store filtered csvs", default="/home/data/preprocessed/")
    parser.add_argument("-s", "--display", action='store_true', default=False)
    parser.add_argument("-r", "--force_redo", action='store_true', default=False)
    parser.add_argument("-p", "--params", action='store', default="params.yaml")
    args = parser.parse_args()

    params = yaml.load(open(args.params, "r"), Loader=yaml.FullLoader)

    sp = SignalPreprocessor(sample_rate=params["frame_rate"])

    if args.force_redo:
        shutil.rmtree(args.output_folder, ignore_errors=True)
        os.makedirs(args.output_folder, exist_ok=True)

    if args.filename is None:
        users = os.listdir(args.signals_directory)
        users = list(filter(lambda x: os.path.isdir(os.path.join(args.signals_directory, x)), users))

        for i, user in enumerate(sorted(users)):
            # get all files
            user_fold = os.path.join(args.signals_directory, user)
            user_files = os.listdir(user_fold)
            user_files = filter(lambda x: os.path.isfile(os.path.join(user_fold, x)), user_files)

            # only retain allowed formats
            user_files = filter(lambda x: x.split(".")[-1] == "csv", user_files)
            os.makedirs(os.path.join(args.output_folder, user), exist_ok=True)

            for file in sorted(user_files):
                filepath = os.path.join(user_fold, file)
                fname = file.split(".")[0]

                csv_fpath = os.path.join(args.output_folder, user, fname + ".csv")
                img_fpath = os.path.join(args.output_folder, user, fname + ".pdf")
                if not os.path.isfile(csv_fpath) or not os.path.isfile(img_fpath):
                    try:

                        to_plot, to_plot_names = [], []

                        extracted_s = pd.read_csv(filepath, index_col=False)
                        preprocessed, columns = [], []
                        sys.stdout.write("{} ({}/{}),{}\n".format(user, i + 1, len(users), file))
                        sys.stdout.flush()
                        for source in params["preprocessor"]["sources"]:
                            assert source in extracted_s.columns.values, "%s not in columns %s" % (source, filepath)
                            for filter_chain in params["preprocessor"]["filter_chains"]:
                                fun_list = filter_chain["flist"]
                                signal_at_step_j = [extracted_s[source].values]
                                name_at_step_j = [source]
                                for j, fun_dict in enumerate(fun_list):
                                    # apply function
                                    fun = getattr(sp, sp.shorter_names[fun_dict["name"]])
                                    filtered_j = fun(
                                        signal_at_step_j[-1],
                                        prev_x=signal_at_step_j[-2] if len(signal_at_step_j)>1 else None,
                                        **fun_dict["params"]
                                    )
                                    new_name = "%s>%s" % (name_at_step_j[-1], fun_dict["name"])

                                    if len(filtered_j) == len(extracted_s[source].values):
                                        signal_at_step_j.append(np.real(filtered_j))  # discard imaginary part if any
                                        name_at_step_j.append(new_name)
                                    else:
                                        to_plot.append(filtered_j)
                                        to_plot_names.append(new_name)
                                preprocessed.extend(signal_at_step_j)
                                columns.extend(name_at_step_j)

                        preprocessed = np.array(preprocessed)
                        assert preprocessed.ndim == 2, "Different functions resulted in different length of preprocessed signal"
                        df = pd.DataFrame(preprocessed.T, columns=columns)
                        df.to_csv(csv_fpath, sep=",", float_format="%.8f", index=False)

                        everything_to_plot = [a.tolist() for a in preprocessed] + to_plot
                        everything_to_plot_labels = columns + to_plot_names

                        visualize_signal(
                            everything_to_plot,
                            labels=columns + to_plot_names,
                            output_fname=img_fpath, )
                    except Exception as e:
                        print(e)


                else:
                    sys.stdout.write("Skipping, file %s exists already\n" % fname)
                    sys.stdout.flush()


