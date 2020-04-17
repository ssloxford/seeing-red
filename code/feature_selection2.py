import argparse
import os

import numpy as np
import pandas as pd
import yaml
from feature_selection1 import FeatureSelector
from sklearn.utils import shuffle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hey it's me")
    parser.add_argument("-o", "--output_folder", help="Directory where to store filtered csvs",
                        default="/home/data/features-selected2/")
    parser.add_argument("-f", "--feature_folder", default="/home/data/features-selected1/")
    parser.add_argument("-s", "--display", action='store_true', default=False)
    parser.add_argument("-r", "--force_redo", action='store_true', default=False)
    parser.add_argument("-p", "--params", action='store', default="params.yaml")

    args = parser.parse_args()

    params = yaml.load(open(args.params, "r"), Loader=yaml.FullLoader)

    fs = FeatureSelector()

    filenames = [
        "features.csv",
        "features-FTA.csv"
    ]

    os.makedirs(args.output_folder, exist_ok=True)

    for fname in filenames:

        fpath = os.path.join(args.feature_folder, fname)
        df = pd.read_csv(fpath, index_col=False)

        # df = hbutils.rows_less_than_n(df, n=60)

        filenames = df["filename"].values
        mcounters = df["meta_counter"].values
        y = df["folder_name"].values
        X = df.drop(["filename", "folder_name", "meta_counter"], axis=1)

        # Create correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

        X = X.drop(to_drop, axis=1)
        print(to_drop, X.columns.values)

        # let's do cross validation for the selection
        # step 1 subsample users
        user_samples_counter = []
        for user in np.unique(y):
            user_samples_counter.append(X[y == user].shape[0])

        min_samples = np.array(user_samples_counter).min()

        X_subset = pd.DataFrame([], columns=X.columns.values)
        y_subset = np.zeros(0, dtype=int)
        for j, user in enumerate(np.unique(y)):
            X_user = X[y == user]
            X_user = shuffle(X_user)
            X_user_sel = X_user[:min_samples]
            X_subset = pd.concat([X_subset, X_user_sel], ignore_index=True)
            y_subset = np.hstack((y_subset, np.ones(min_samples, dtype=int)*j))

        selected_for_fun = []
        for function in params["feature_selection2"]:
            fun_name = function["name"]
            fun_params = function["params"]
            fun = getattr(fs, fun_name)
            X_subset_selected = fun(X_subset, X_subset, y_subset, X_subset.columns.values, **fun_params)
            selected_for_fun.append(set(X_subset_selected.columns.values))

        feature_intersection = selected_for_fun[0]
        for j, function in enumerate(params["feature_selection2"][1:]):
            feature_intersection = feature_intersection & selected_for_fun[j+1]

        feature_intersection = list(feature_intersection)

        X_selected = X[feature_intersection].copy()

        X_selected["filename"] = filenames
        X_selected["meta_counter"] = mcounters
        X_selected["folder_name"] = y

        output_file = os.path.join(args.output_folder, fname)
        X_selected.to_csv(output_file, index=False)








