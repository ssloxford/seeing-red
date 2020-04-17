import argparse
import os

import hb_utils as hbutils
import numpy as np
import pandas as pd
import pymrmr
import yaml
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


class FeatureSelector():

    def __init__(self, ):
        pass

    def scale(self, scaler, X_train, X_test):
        scaler.fit(X_train)
        out = scaler.transform(X_test)
        return out, scaler

    def pca(self, X_train, X_test, y_train, feat_names, **kwargs):
        n_components = kwargs["n_components"]
        scale = kwargs["scale"]
        scaler = kwargs["scaler"]
        explained_v = kwargs["explained_v"]
        prefix = kwargs["prefix"]
        pca = PCA(n_components=n_components)
        X_train_scaled = X_train
        if scale:
            X_train_scaled = self.scale(scaler, X_train, X_train)
        pca.fit(X_train_scaled)

        # check how many components we need to get explained_v
        ex_v_ = np.cumsum(pca.explained_variance_/pca.explained_variance_.sum())
        n_comp_retain = np.argwhere(ex_v_>=explained_v).flatten()[0]+1
        transformed = pca.transform(X_test)[:, :n_comp_retain]
        transformed = pd.DataFrame(transformed, columns=["%s_pca_%d" % (prefix, x) for x in range(n_comp_retain)])
        return transformed

    def jitter(self, a_series, noise_reduction=1000000):
        return (np.random.random(len(a_series)) * a_series.std() / noise_reduction) - (
                    a_series.std() / (2 * noise_reduction))

    def mRMR(self, X_train, X_test, y_train, feat_names, **kwargs):
        outliers = kwargs["outliers"]
        n_bins = kwargs["n_bins"]
        method = kwargs["method"]
        retain_ratio = kwargs["retain_ratio"]
        top_n = int(retain_ratio*len(feat_names))
        if top_n is None:
            top_n = X_train.shape[1]
        if y_train.dtype != int:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train).astype(int)
        feat_names = list(feat_names)
        df = pd.DataFrame(np.hstack((y_train[:, np.newaxis], X_train)), columns=["label"] + feat_names)
        df_bin = df.copy()
        for f in feat_names:
            series = df[f]
            if outliers:
                # remove outliers binning in 1st<>99th percentile
                if not np.all(series.values == series.values[0]):
                    # only do this step when series is made by at least 2 different values, otherwise something crashes
                    _, bins = pd.qcut(series+self.jitter(series), np.linspace(0, 1, 100), retbins=True)
                    first_perc, ninetyninth_perc = bins[0], bins[-1]
                    series = np.maximum(series, first_perc)
                    series = np.minimum(series, ninetyninth_perc)
            df_bin[f] = pd.cut(series, bins=n_bins, labels=np.arange(0, n_bins))
        which_features = pymrmr.mRMR(df_bin, method, top_n)
        return df[which_features]

    def rmi(self, X_train, X_test, y_train, feat_names, min_rmi=None, retain_ratio=None, **kwargs):
        top_n = int(retain_ratio*len(feat_names))
        if y_train.dtype != int:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train).astype(int)

        entropy = self.discrete_entropy(y_train)
        rmi = np.array([i / entropy for i in list(feature_selection.mutual_info_classif(X_train, y_train, random_state=0))])

        sorted_rmi_i = np.argsort(rmi)[::-1]
        rmi = rmi[sorted_rmi_i]
        feat_names_sorted = np.array(feat_names)[sorted_rmi_i].tolist()

        indexes_to_retain = np.argwhere(rmi>=min_rmi).flatten() if min_rmi is not None else range(top_n)

        which_features = [feat_names_sorted[i] for i in indexes_to_retain]
        df = X_train.copy()

        print("### RMI ###")
        for i, feat in enumerate(feat_names_sorted):
            print(feat, rmi[i])

        return df[which_features]

    def discrete_entropy(self, y):
        bincount = np.bincount(y)
        nonzero_bins = bincount[bincount > 0].astype(float)
        probabilities = nonzero_bins / float(nonzero_bins.sum())
        return -(probabilities * np.log(probabilities)).sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey it's me")
    parser.add_argument("-o", "--output_folder", help="Directory where to store filtered csvs",
                        default="/home/data/features-selected1/")
    parser.add_argument("-f", "--feature_folder", default="/home/data/features/")
    parser.add_argument("-s", "--display", action='store_true', default=False)
    parser.add_argument("-r", "--force_redo", action='store_true', default=False)
    parser.add_argument("-p", "--params", action='store', default="params.yaml")

    args = parser.parse_args()

    params = yaml.load(open(args.params, "r"), Loader=yaml.FullLoader)

    fs = FeatureSelector()

    # let's do the extraction twice in order to save both
    # FTA-filtered samples and non-FTA-filtered ones

    ffile_postfixes = ["", "-FTA"]

    os.makedirs(args.output_folder, exist_ok=True)

    for postfix in ffile_postfixes:

        hbutils.delete_all_subdirs(args.output_folder)

        # get all feature groups
        files = hbutils.get_all_fmt_files("/home/data/features/", "csv")
        feature_groups = set(map(lambda x: x.split(".")[0].split("-")[0], files))

        files_to_append = []

        for fg in feature_groups:
            feat_fpath = os.path.join(args.feature_folder, fg+postfix+".csv")
            output_file = os.path.join(args.output_folder, fg+postfix+".csv")
            df = pd.read_csv(feat_fpath, index_col=False)
            # df = hbutils.rows_less_than_n(df, n=60)
            y = df["folder_name"].values
            filenames = df["filename"].values
            mcounters = df["meta_counter"].values
            X_selected = df.copy()

            if fg in params["feature_selection1"]:
                X = df.drop(["filename", "folder_name", "meta_counter"], axis=1)
                fun_name = params["feature_selection1"][fg]["name"]
                fun = getattr(fs, fun_name)
                X_selected = fun(X, X, y, X.columns.values, **params["feature_selection1"][fg]["params"])
                X_selected["filename"] = filenames
                X_selected["folder_name"] = y

            X_selected.to_csv(output_file, index=False)
            files_to_append.append(output_file)

        hbutils.merge_csv_horizontal(files_to_append, os.path.join(args.output_folder, "features"+postfix+".csv"))








