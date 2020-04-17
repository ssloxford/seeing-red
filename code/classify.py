import argparse
import json
import os
import random
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats
import xgboost as xgb
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

matplotlib.use("agg")
import matplotlib.patches as mpatches
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import hb_utils as hbutils
import yaml
from joblib import Parallel, delayed

random.seed(0)

boxprops = {
    "color": "black",
    "linestyle": "-",
}

whiskerprops = {
    "linestyle": "-",
    "color": "black"
}
capprops = dict(linestyle='-', linewidth=1, color='black')

medianprops = dict(linestyle='-', linewidth=3, color='firebrick')

colors = ['lightblue', 'pink', "red", "blue", "green", "cyan", "yellow", "orange", "gray"]

hatches = ["\\\\", "//"]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def exp1(df, aggr_wind_size, is_fta):
    """
    Multi-class classification shuffled
    """
    classifiers = [
        svm.SVC(kernel="rbf", C=1.0, gamma="auto", probability=True),
        xgb.XGBClassifier(),
        RandomForestClassifier(n_estimators=50)
    ]

    aggr_functions = [np.mean, np.mean, np.mean]

    scale_inputs = [True, False, False, False, False]

    n_folds = 2

    skfold = StratifiedKFold(n_splits=n_folds)

    ss = preprocessing.StandardScaler()

    X = df.drop(["filename", "meta_counter", "folder_name"], axis=1).values
    y = df["folder_name"].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    big_result = np.zeros(shape=(n_folds, len(classifiers), len(aggr_wind_size), len(np.unique(y))))

    for i0, (train_i, test_i) in enumerate(skfold.split(X, y)):

        for i1, clf in enumerate(classifiers):

            X_train, X_test = X[train_i], X[test_i]
            y_train, y_test = y[train_i], y[test_i]

            if scale_inputs[i1]:
                ss.fit(X[train_i])
                X_train = ss.transform(X_train)
                X_test = ss.transform(X_test)

            clf.fit(X_train, y_train)
            for i3, user in enumerate(np.sort(np.unique(y))):
                X_user, X_others = X_test[y_test == user], X_test[y_test != user]
                y_others = y_test[y_test != user]

                user_dists = clf.predict_proba(X_user)[:, user]
                other_dists = clf.predict_proba(X_others)[:, user]

                for i2, aggr_size in enumerate(aggr_wind_size):
                    user_dists_agg = aggr(user_dists, aggr_size, reps=100, function=aggr_functions[i1])
                    other_dists_agg = []
                    for other_u in np.unique(y_others):
                        assert other_u != user
                        _other_dists = other_dists[y_others == other_u]
                        new_aggr = aggr(_other_dists, aggr_size, reps=10, function=aggr_functions[i1])
                        other_dists_agg.extend(new_aggr)

                    y_true = np.concatenate((np.ones(len(user_dists_agg)), np.zeros(len(other_dists_agg))), axis=0)
                    y_pred = np.concatenate((user_dists_agg, other_dists_agg), axis=0)

                    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
                    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                    thresh = interp1d(fpr, thresholds)(eer)
                    big_result[i0, i1, i2, i3] = eer

    basename = "all" + ("-fta" if is_fta else "")
    of = os.path.join(output_folder, "exp1/")
    config = {
        "header": ["fold", "clf", "window_size", "user"],
        "fold": [0, 1],
        "clf": ['SVM', 'GBT', "RF"],
        "window_size": aggr_wind_size,
        "user": list(map(str, df["folder_name"].unique()))}
    os.makedirs(of, exist_ok=True)
    np.save(os.path.join(of, "%s.npy" % basename), big_result)
    if basename == "all":
        json.dump(config, open(os.path.join(of, "descr.json"), "w"), indent=2, )

    return big_result


def aggr(dists, size, reps, function):
    dists_agg = []
    for j in range(reps):
        _ud = shuffle(dists)
        # print(_ud[:size])
        _ud_median = function(_ud[:size])
        dists_agg.append(_ud_median)
    return dists_agg


def get_cons_samples(csdf, fnames, classifier, scaler, aggrsize, aggrfun, scale_input):
    distances = []
    for _fname in fnames:
        # select samples from this file
        _xtest = csdf[csdf["filename"] == _fname].copy()
        _xtest = _xtest.drop(["filename", "meta_counter", "folder_name"], axis=1).values
        if scale_input:
            _xtest = scaler.transform(_xtest)
        _udists = classifier.decision_function(_xtest)
        distances.extend(aggr(_udists, aggrsize, reps=10, function=aggrfun))
    return distances


def exp3(classifiers, df, number_of_enrolments, ns_of_train_data, aggr_sizes, aggr_functions, scale_inputs, is_fta):
    """
    One class classification shuffled
    """

    labels = df["folder_name"]
    X = df.drop(["filename", "meta_counter", "folder_name"], axis=1).values

    le = preprocessing.LabelEncoder()
    le.fit(labels.values)
    y = le.transform(labels.values)

    X, y = shuffle(X, y)

    funs = [lambda x: (np.e**x/(np.e**x+1)+1)/2.0, lambda x: (np.e**x/(np.e**x+1)+1)/2.0]
    funs = [lambda x: x, lambda x: x]
    labels = np.sort(np.unique(y))

    assert len(classifiers) == len(aggr_functions) == len(scale_inputs)
    ss = preprocessing.StandardScaler()
    big_res_auc = np.zeros(shape=(len(ns_of_train_data), number_of_enrolments, len(aggr_sizes), len(labels), len(classifiers)+1))
    big_eer = np.zeros(shape=(len(ns_of_train_data), number_of_enrolments, len(aggr_sizes), len(labels), len(classifiers)+1))

    for i1, tp in enumerate(ns_of_train_data):

        for i2 in range(number_of_enrolments):

            sys.stdout.write("\r%d/%d" % (i1*number_of_enrolments+i2, len(ns_of_train_data)*number_of_enrolments))
            sys.stdout.flush()

            for i3, size in enumerate(aggr_sizes):

                for i4, user in enumerate(labels):

                    all_neg_dists = {h: [] for h, _ in enumerate(classifiers)}
                    all_pos_dists = {h: [] for h, _ in enumerate(classifiers)}

                    X_user = X[y == user]
                    X_others, y_others = X[y != user], y[y != user]
                    X_user_shuffled = shuffle(X_user)
                    n_train = tp
                    X_u_train, X_u_test = X_user_shuffled[:n_train], X_user_shuffled[n_train:]
                    X_u_train_c = X_u_train
                    X_u_test_c = X_u_test
                    X_others_c = X_others

                    for i5, clf in enumerate(classifiers):

                        if scale_inputs[i5]:
                            ss.fit(X)
                            X_u_train_c = ss.transform(X_u_train)
                            X_u_test_c = ss.transform(X_u_test)
                            X_others_c = ss.transform(X_others)

                        clf.fit(X_u_train_c)
                        user_dists = clf.decision_function(X_u_test_c)
                        other_dists = clf.decision_function(X_others_c)

                        user_dists_agg = aggr(user_dists, size, reps=50, function=aggr_functions[i5])
                        other_dists_agg = []
                        for other_u in np.unique(y_others):
                            _other_dists = other_dists[y_others == other_u]
                            new_medians = aggr(_other_dists, size, reps=20, function=aggr_functions[i5])
                            other_dists_agg.extend(new_medians)

                        all_neg_dists[i5].extend(other_dists_agg)
                        all_pos_dists[i5].extend(user_dists_agg)

                    # now we want a table of (n, n_classifiers) with one column per classifier with distances
                    big_distances = np.zeros(shape=(len(all_neg_dists[0])+len(all_pos_dists[0]), len(classifiers)))
                    y_true = np.concatenate((np.ones(len(all_pos_dists[0])), np.zeros(len(all_neg_dists[0]))), axis=0)
                    for i5, clf in enumerate(classifiers):
                        _all_dists = all_pos_dists[i5] + all_neg_dists[i5]
                        big_distances[:, i5] = _all_dists

                    prep_for_comb = [[], []]
                    for i5, clf in enumerate(classifiers):
                        y_pred = big_distances[:, i5]

                        prep_for_comb[i5] = [funs[i5](x) for x in big_distances[:, i5]]
                        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
                        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                        thresh = interp1d(fpr, thresholds)(eer)
                        big_eer[i1, i2, i3, i4, i5] = eer
                        big_res_auc[i1, i2, i3, i4, i5] = metrics.roc_auc_score(y_true, y_pred)

                    y_pred = [x+y for x, y in zip(prep_for_comb[0], prep_for_comb[1])]
                    i5 = len(classifiers)
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
                    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                    thresh = interp1d(fpr, thresholds)(eer)
                    big_eer[i1, i2, i3, i4, i5] = eer
                    big_res_auc[i1, i2, i3, i4, i5] = metrics.roc_auc_score(y_true, y_pred)

    basename = "all" + ("-fta" if is_fta else "")
    of = os.path.join(output_folder, "exp3/")
    config = {
        "header": ["no_train_data", "no_enrol", "window_size", "user", "clf"],
        "no_train_data": ns_of_train_data,
        "no_enrol": list(range(number_of_enrolments)),
        "window_size": aggr_wind_size,
        "clf": ['OSVM', 'IsFrst', "OSVM+IsFrst"],
        "user": list(map(str, df["folder_name"].unique()))}
    os.makedirs(of, exist_ok=True)
    np.save(os.path.join(of, "%s.npy" % basename), big_eer)
    if basename == "all":
        json.dump(config, open(os.path.join(of, "descr.json"), "w"), indent=2, )

    return big_eer, big_res_auc


def exp4(classifiers, df, number_of_enrolments, ns_of_train_data, aggr_sizes, aggr_functions, scale_inputs, is_fta):
    """
    Sequential one class classification
    """
    assert len(classifiers) == len(aggr_functions) == len(scale_inputs)
    assert "folder_name" in df.columns.values
    assert "meta_counter" in df.columns.values
    assert "filename" in df.columns.values

    labels = list(sorted(df["folder_name"].unique()))
    print(labels)
    funs = [lambda x: x, lambda x: x]

    ss = preprocessing.StandardScaler()
    big_res_auc = np.zeros(shape=(len(ns_of_train_data), number_of_enrolments, len(aggr_sizes), len(labels), len(classifiers) + 1))
    big_eer = np.zeros(shape=(len(ns_of_train_data), number_of_enrolments, len(aggr_sizes), len(labels), len(classifiers) + 1))

    for i1, tp in enumerate(ns_of_train_data):

        for i2 in range(number_of_enrolments):

            sys.stdout.write("\r%d/%d" % (i1*number_of_enrolments+i2, len(ns_of_train_data)*number_of_enrolments))
            sys.stdout.flush()

            for i3, size in enumerate(aggr_sizes):

                for i4, user in enumerate(labels):

                    all_neg_dists = {h: [] for h, _ in enumerate(classifiers)}
                    all_pos_dists = {h: [] for h, _ in enumerate(classifiers)}

                    u_df = df[df["folder_name"] == user].copy()
                    usr_fnames = pd.unique(u_df["filename"].values)

                    # choose tp files for training and remaining for testing
                    usr_fnames_train = np.random.choice(usr_fnames, size=len(usr_fnames)-1, replace=False).tolist()
                    usr_fnames_test = list(set(usr_fnames) - set(usr_fnames_train))

                    # filter dataframe with test/train files
                    u_df_train = df[df["filename"].isin(usr_fnames_train)]
                    u_df_test = df[df["filename"].isin(usr_fnames_test)]
                    o_df = df[df["folder_name"] != user].copy()

                    for i5, clf in enumerate(classifiers):

                        X_train = u_df_train.drop(["filename", "meta_counter", "folder_name"], axis=1)
                        X_others = o_df.drop(["filename", "meta_counter", "folder_name"], axis=1)

                        if scale_inputs[i5]:
                            ss.fit(np.vstack((X_train, X_others)))
                            X_train = ss.transform(X_train)

                        clf.fit(X_train)

                        user_dists_agg = get_cons_samples(u_df_test, usr_fnames_test, clf, ss, size, aggr_functions[i5], scale_inputs[i5])
                        other_dists_agg = []

                        # fill with negative samples
                        for other_u in o_df["folder_name"].unique():
                            # list filenames for this user
                            other_u_df = o_df[o_df["folder_name"] == other_u].copy()
                            # make a list with other_u filenames
                            this_user_fnames = shuffle(other_u_df["filename"].unique())[:10]
                            other_dists_agg.extend(get_cons_samples(other_u_df, this_user_fnames, clf, ss, size, aggr_functions[i5], scale_inputs[i5]))

                        all_neg_dists[i5].extend(other_dists_agg)
                        all_pos_dists[i5].extend(user_dists_agg)

                    # now we want a table of (n, n_classifiers) with one column per classifier with distances
                    big_distances = np.zeros(shape=(len(all_neg_dists[0])+len(all_pos_dists[0]), len(classifiers)))
                    y_true = np.concatenate((np.ones(len(all_pos_dists[0])), np.zeros(len(all_neg_dists[0]))), axis=0)

                    for i5, clf in enumerate(classifiers):
                        _all_dists = all_pos_dists[i5] + all_neg_dists[i5]
                        big_distances[:, i5] = _all_dists

                    prep_for_comb = [[], []]
                    for i5, clf in enumerate(classifiers):
                        y_pred = big_distances[:, i5]

                        prep_for_comb[i5] = [funs[i5](x) for x in big_distances[:, i5]]
                        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
                        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                        thresh = interp1d(fpr, thresholds)(eer)
                        big_eer[i1, i2, i3, i4, i5] = eer
                        big_res_auc[i1, i2, i3, i4, i5] = metrics.roc_auc_score(y_true, y_pred)

                    y_pred = [x + y for x, y in zip(prep_for_comb[0], prep_for_comb[1])]
                    i5 = len(classifiers)
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
                    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                    thresh = interp1d(fpr, thresholds)(eer)
                    big_eer[i1, i2, i3, i4, i5] = eer
                    big_res_auc[i1, i2, i3, i4, i5] = metrics.roc_auc_score(y_true, y_pred)

    basename = "all" + ("-fta" if is_fta else "")
    of = os.path.join(output_folder, "exp4/")
    config = {
        "header": ["no_train_data", "no_enrol", "window_size", "user", "clf"],
        "no_train_data": ns_of_train_data,
        "no_enrol": list(range(number_of_enrolments)),
        "window_size": aggr_wind_size,
        "clf": ['OSVM', 'IsFrst', "OSVM+IsFrst"],
        "user": list(map(str, df["folder_name"].unique()))}
    os.makedirs(of, exist_ok=True)
    np.save(os.path.join(of, "%s.npy" % basename), big_eer)
    if basename == "all":
        json.dump(config, open(os.path.join(of, "descr.json"), "w"), indent=2, )

    return big_eer, big_res_auc


def exp5(classifiers, df, number_of_enrolments, ns_of_train_data, aggr_sizes, aggr_functions, scale_inputs):
    """
    Sequential multi-class classification, attacker is in negative set
    """

    assert len(classifiers) == len(aggr_functions) == len(scale_inputs)
    assert "folder_name" in df.columns.values
    assert "meta_counter" in df.columns.values
    assert "filename" in df.columns.values

    labels = df["folder_name"]
    unique_labels = labels.unique()

    ss = preprocessing.StandardScaler()
    big_res_auc = np.zeros(shape=(len(classifiers), len(ns_of_train_data), number_of_enrolments, len(aggr_sizes), len(unique_labels)))
    big_eer = np.zeros(shape=(len(classifiers), len(ns_of_train_data), number_of_enrolments, len(aggr_sizes), len(unique_labels)))

    for i0, clf in enumerate(classifiers):

        for i1, tp in enumerate(ns_of_train_data):

            for i2 in range(number_of_enrolments):

                sys.stdout.write("\r%d/%d" % (i1*number_of_enrolments+i2, len(ns_of_train_data)*number_of_enrolments))
                sys.stdout.flush()

                for i3, size in enumerate(aggr_sizes):

                    for i4, user in enumerate(unique_labels):

                        u_df = df[df["folder_name"] == user].copy()
                        usr_fnames = pd.unique(u_df["filename"].values)
                        no_train_files_user = 2
                        no_train_files_others = 5
                        # choose tp files for training and remaining for testing
                        usr_fnames_train = np.random.choice(usr_fnames, size=no_train_files_user, replace=False).tolist()
                        usr_fnames_test = list(set(usr_fnames) - set(usr_fnames_train))
                        assert len(set(usr_fnames_train) & set(usr_fnames_test)) == 0
                        # filter dataframe with test/train files
                        u_df_train = u_df[u_df["filename"].isin(usr_fnames_train)]
                        u_df_test = u_df[u_df["filename"].isin(usr_fnames_test)]
                        o_df = df[df["folder_name"] != user].copy()

                        X_train_user = u_df_train.drop(["filename", "meta_counter", "folder_name"], axis=1)
                        X_train_others = np.zeros(shape=(0, X_train_user.shape[1]))
                        others_fnames_train = dict()
                        others_fnames_test = dict()

                        for other_u in o_df["folder_name"].unique():
                            assert other_u != user
                            # list filenames for this user
                            other_u_df = o_df[o_df["folder_name"] == other_u].copy()
                            # make a list with other_u filenames
                            this_user_fnames = shuffle(other_u_df["filename"].unique())
                            # select no_train_files_others filenames for train and remanining for test
                            this_user_fnames_train = this_user_fnames[:no_train_files_others].tolist()
                            this_user_fnames_test = this_user_fnames[no_train_files_others:].tolist()
                            assert len(set(this_user_fnames_train) & set(this_user_fnames_test)) == 0
                            others_fnames_train[other_u] = this_user_fnames_train
                            others_fnames_test[other_u] = this_user_fnames_test
                            # now select those selected fnames
                            other_u_df = other_u_df[other_u_df["filename"].isin(this_user_fnames_train)].copy()
                            X_train_others = np.vstack(
                                (X_train_others, other_u_df.drop(["filename", "meta_counter", "folder_name"], axis=1)))

                        # at the end of this loop, X_train_user contains user train data from no_train_files_user files
                        # while X_train_others contains every other users train data from no_train_files_others files
                        # each now fit the classifier
                        y_user = np.ones(X_train_user.shape[0])
                        y_others = np.zeros(X_train_others.shape[0])

                        X_train = np.vstack((X_train_user, X_train_others))
                        y_train = np.hstack((y_user, y_others))
                        if scale_inputs[i0]:
                            ss.fit(X_train)
                            X_train = ss.transform(X_train)
                        clf = xgb.XGBClassifier()
                        clf.fit(X_train, y_train)

                        # now we do the aggregation bit
                        user_dists_agg = get_cons_samples(
                            u_df_test, usr_fnames_test, clf, ss, size, aggr_functions[i0], scale_inputs[i0])
                        other_dists_agg = []

                        for other_u in o_df["folder_name"].unique():
                            # use the previously defined dictionary to fetch testing files for this other_user
                            other_u_df = o_df[o_df["folder_name"] == other_u].copy()
                            other_dists_agg.extend(get_cons_samples(
                                other_u_df, others_fnames_test[other_u], clf, ss, size, aggr_functions[i0], scale_inputs[i0]))

                        y_true = np.concatenate((np.ones(len(user_dists_agg)), np.zeros(len(other_dists_agg))), axis=0)
                        y_pred = np.concatenate((user_dists_agg, other_dists_agg), axis=0)

                        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
                        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                        thresh = interp1d(fpr, thresholds)(eer)
                        big_eer[i0, i1, i2, i3, i4] = eer
                        big_res_auc[i0, i1, i2, i3, i4] = metrics.roc_auc_score(y_true, y_pred)

    return big_eer, big_res_auc


class AnyObjectHandler(object):
    def __init__(self, hatch="xx", facecolor='red', lw=1, edgecolor="black"):
        self._hatch = hatch
        self._facecolor = facecolor
        self._lw = lw
        self._edgecolor = edgecolor

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor=self._facecolor,
                                   edgecolor=self._edgecolor, hatch=self._hatch, lw=self._lw,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


def create_classifiers(classifiers_list):
    cmap = {
        "svm": svm.OneClassSVM,
        "isolationforest": IsolationForest,
    }

    aggr_solver = {
        "mean": np.mean,
        "median": np.median
    }

    classifiers = []

    aggr_method = []

    for classifier in classifiers_list:
        cname = classifier["name"].split("_")[0]
        clf = cmap[cname](**classifier["params"])
        classifiers.append(clf)
        aggr_method.append(aggr_solver[classifier["aggregation"]])
    return classifiers, aggr_method


def do_one(i2, i1, aggr_sizes, classifiers, df, big_eer, big_res_auc, labels, scale_inputs, ss, funs):
    print(i1, i2)
    for i3, size in enumerate(aggr_sizes):
        for i4, user in enumerate(labels):
            all_neg_dists = {h: [] for h, _ in enumerate(classifiers)}
            all_pos_dists = {h: [] for h, _ in enumerate(classifiers)}
            u_df = df[df["folder_name"] == user].copy()
            usr_fnames = pd.unique(u_df["filename"].values)
            # choose tp files for training and remaining for testing
            usr_fnames_train = np.random.choice(usr_fnames, size=len(usr_fnames) - 1, replace=False).tolist()
            usr_fnames_test = list(set(usr_fnames) - set(usr_fnames_train))
            # filter dataframe with test/train files
            u_df_train = df[df["filename"].isin(usr_fnames_train)]
            u_df_test = df[df["filename"].isin(usr_fnames_test)]
            o_df = df[df["folder_name"] != user].copy()
            for i5, clf in enumerate(classifiers):
                X_train = u_df_train.drop(["filename", "meta_counter", "folder_name"], axis=1)
                X_others = o_df.drop(["filename", "meta_counter", "folder_name"], axis=1)
                if scale_inputs[i5]:
                    ss.fit(np.vstack((X_train, X_others)))
                    X_train = ss.transform(X_train)
                clf.fit(X_train)
                user_dists_agg = get_cons_samples(u_df_test, usr_fnames_test, clf, ss, size, aggr_functions[i5],
                                                  scale_inputs[i5])
                other_dists_agg = []
                # fill with negative samples
                for other_u in o_df["folder_name"].unique():
                    # list filenames for this user
                    other_u_df = o_df[o_df["folder_name"] == other_u].copy()
                    # make a list with other_u filenames
                    this_user_fnames = shuffle(other_u_df["filename"].unique())[:10]
                    other_dists_agg.extend(
                        get_cons_samples(other_u_df, this_user_fnames, clf, ss, size, aggr_functions[i5],
                                         scale_inputs[i5]))

                all_neg_dists[i5].extend(other_dists_agg)
                all_pos_dists[i5].extend(user_dists_agg)

            # now we want a table of (n, n_classifiers) with one column per classifier with distances
            big_distances = np.zeros(shape=(len(all_neg_dists[0]) + len(all_pos_dists[0]), len(classifiers)))
            y_true = np.concatenate((np.ones(len(all_pos_dists[0])), np.zeros(len(all_neg_dists[0]))), axis=0)

            for i5, clf in enumerate(classifiers):
                _all_dists = all_pos_dists[i5] + all_neg_dists[i5]
                big_distances[:, i5] = _all_dists

            prep_for_comb = [[], []]
            for i5, clf in enumerate(classifiers):
                y_pred = big_distances[:, i5]

                prep_for_comb[i5] = [funs[i5](x) for x in big_distances[:, i5]]
                fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
                eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                thresh = interp1d(fpr, thresholds)(eer)
                big_eer[i1, i2, i3, i4, i5] = eer
                big_res_auc[i1, i2, i3, i4, i5] = metrics.roc_auc_score(y_true, y_pred)

            y_pred = [x + y for x, y in zip(prep_for_comb[0], prep_for_comb[1])]
            i5 = len(classifiers)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            thresh = interp1d(fpr, thresholds)(eer)
            big_eer[i1, i2, i3, i4, i5] = eer
            big_res_auc[i1, i2, i3, i4, i5] = metrics.roc_auc_score(y_true, y_pred)
    big_eer.flush()
    big_res_auc.flush()


def exp4_parallel(classifiers, df, number_of_enrolments, ns_of_train_data, aggr_sizes, aggr_functions, scale_inputs, is_fta):
    """
    Sequential one class classification
    """
    assert len(classifiers) == len(aggr_functions) == len(scale_inputs)
    assert "folder_name" in df.columns.values
    assert "meta_counter" in df.columns.values
    assert "filename" in df.columns.values

    labels = list(sorted(df["folder_name"].unique()))
    print(labels)
    funs = [lambda x: x, lambda x: x]

    ss = preprocessing.StandardScaler()

    of = os.path.join(output_folder, "exp4/")
    os.makedirs(of, exist_ok=True)
    basename = "all" + ("-fta" if is_fta else "")

    big_res_auc = np.memmap(os.path.join(of, "%s-dumb.npy" % basename), dtype="float32", mode="w+", shape=(
    len(ns_of_train_data), number_of_enrolments, len(aggr_sizes), len(labels), len(classifiers) + 1))
    big_eer = np.memmap(os.path.join(of, "%s.npy" % basename), dtype="float32", mode="w+", shape=(
    len(ns_of_train_data), number_of_enrolments, len(aggr_sizes), len(labels), len(classifiers) + 1))

    all_clfiers = []
    for i2 in range(number_of_enrolments):
        for i1 in ns_of_train_data:
            clfs, _ = create_classifiers(params["classification"]["classifiers"])
            all_clfiers.append(clfs)

    Parallel(n_jobs=32)(delayed(do_one)(
        i2, i1, aggr_sizes, all_clfiers[i1 * i2 + i1], df.copy(), ns_of_train_data, big_eer, big_res_auc, labels,
        scale_inputs, ss, funs) for i2 in range(number_of_enrolments) for i1 in range(len(ns_of_train_data)))

    config = {
        "header": ["no_train_data", "no_enrol", "window_size", "user", "clf"],
        "no_train_data": ns_of_train_data,
        "no_enrol": list(range(number_of_enrolments)),
        "window_size": aggr_wind_size,
        "clf": ['OSVM', 'IsFrst', "OSVM+IsFrst"],
        "user": list(map(str, df["folder_name"].unique()))}
    if basename == "all":
        json.dump(config, open(os.path.join(of, "descr.json"), "w"), indent=2, )

    return big_eer, big_res_auc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ne", "--no_of_enrol", default=10, type=int, help="No. of enrolment repetitions")
    parser.add_argument("-r", "--force_redo", action='store_true', default=False)
    parser.add_argument("-p", "--params", action='store', default="params.yaml")

    global output_folder
    output_folder = "/home/data/results"
    os.makedirs(output_folder, exist_ok=True)

    args = parser.parse_args()

    feature_filenames = [
        "features.csv",
        "features-FTA.csv"
    ]

    params = yaml.load(open(args.params, "r"), Loader=yaml.FullLoader)

    classifiers, aggr_functions = create_classifiers(params["classification"]["classifiers"])

    aggr_wind_size = [1, 2, 5, 10, 20]
    tr_data_amount = [10, 20, 30]
    n_files_training = [1, 2, 3]

    for i, ff in enumerate(feature_filenames):

        df = pd.read_csv(os.path.join("/home/data/features-selected2/", ff), index_col=False, )

        df = hbutils.rows_less_than_n(df, n=60)

        print("running Multi class (exp1)")
        _ = exp1(df, aggr_wind_size, "FTA" in ff)
        print("running One class (exp3)")
        _, _ = exp3(classifiers, df, args.no_of_enrol, tr_data_amount, aggr_wind_size,
                             aggr_functions=aggr_functions, scale_inputs=[True, False], is_fta="FTA" in ff)
        # big_eer, big_auc = exp5([xgb.XGBClassifier()], df, args.no_of_enrol, n_files_training, aggr_wind_size,
        #                         aggr_functions=[np.mean], scale_inputs=[True])
        print("running One class Cross-session (exp4)")
        _, _ = exp4(classifiers, df, args.no_of_enrol, n_files_training, aggr_wind_size,
                                aggr_functions=aggr_functions, scale_inputs=[True, False], is_fta="FTA" in ff)
        # print_stuff(big_eer.mean(axis=0), aggr_wind_size, n_files_training, df["folder_name"].unique())
        # print_stuff(big_eer, aggr_wind_size, tr_data_amount, df["folder_name"].unique())

        #exit()

