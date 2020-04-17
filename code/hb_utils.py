from scipy import interpolate
import numpy as np
import os
import pandas as pd
import shutil
from sklearn.preprocessing import LabelEncoder


def derivative(signal, index=1):
    d1 = np.array([0] + [b - a for a, b in zip(signal, signal[1:])])
    if index == 1:
        return d1
    elif index == 2:
        return np.array([0] + [b - a for a, b in zip(d1, d1[1:])])
    else:
        raise ValueError("Only support first or second derivatives")


def rows_less_than_n(df, label_column="folder_name", n=60):
    retdf = df.copy()
    y_l = df[label_column]
    le = LabelEncoder()
    y = le.fit_transform(y_l)
    bc = np.bincount(y)
    for i, item in enumerate(bc):
        if bc[i]<n:
            # print("Removing class %s with %d samples" % (le.classes_[i], bc[i]))
            retdf = retdf.drop(retdf[retdf.folder_name.values == le.classes_[i]].index)
    return retdf


def merge_features_into_file(directory, postfix):
    """
    Take all feature files and merge them together
    :param directory:
    :param out_fname:
    :return:
    """
    feat_subf = os.listdir(directory)
    feat_subf = list(filter(lambda x: os.path.isdir(os.path.join(directory, x)), feat_subf))

    users = os.listdir(os.path.join(directory, feat_subf[0]))
    users = list(filter(lambda x: os.path.isdir(os.path.join(directory, feat_subf[0], x)), users))

    for fsf in feat_subf:
        for user in users:
            merge_all_csv_into_one(os.path.join(directory, fsf, user))

    for fsf in feat_subf:
        merge_all_csv_into_one(os.path.join(directory, fsf), postfix=postfix)

    print(users, feat_subf)


def get_all_fmt_files(folder, fmt="csv"):
    files = os.listdir(folder)
    files = list(filter(lambda x: os.path.isfile(os.path.join(folder, x)), files))
    # only retain allowed formats
    files = list(filter(lambda x: x.split(".")[-1] == fmt, files))
    return files


def merge_csv_horizontal(fpaths, outfilepath):
    big_df = pd.read_csv(fpaths[0], index_col=None)
    for fpath in fpaths[1:]:
        big_df = big_df.join(pd.read_csv(fpath, index_col=None), lsuffix="", rsuffix="-removeme")

    col_names = set(big_df.columns.values)

    for col in col_names:
        if "-removeme" in col:
            big_df = big_df.drop(col, axis=1)

    big_df.to_csv(outfilepath, index=False)
    return True


def merge_all_csv_into_one(directory, keep_folder_name=True, keep_filename=True, postfix=""):
    output = []
    parent_dir = os.sep.join(directory.split(os.sep)[:-1])
    files_to_merge = get_all_fmt_files(directory, fmt="csv")
    for f in files_to_merge:
        filepath = os.path.join(directory, f)
        folder_name = directory.split(os.sep)[-1]
        fname = f.split(".")[0]

        df = pd.read_csv(filepath, index_col=False)
        columns = df.columns.values.tolist()
        this_mat = df.values
        for j in range(this_mat.shape[0]):
            row = []
            if keep_filename and "filename" not in columns:
                row += [fname]
            if keep_folder_name and "folder_name" not in columns:
                row += [folder_name]

            row += this_mat[j].tolist()
            output.append(row)

    col_names = []
    if keep_filename and "filename" not in columns:
        col_names += ["filename"]
    if keep_folder_name and "folder_name" not in columns:
        col_names += ["folder_name"]

    col_names += columns

    df = pd.DataFrame(output, columns=col_names)
    df.to_csv(os.path.join(parent_dir, folder_name + postfix + ".csv"), index=False)


def delete_all_subdirs(base_dir):
    for subdir in os.listdir(base_dir):
        subd_fpath = os.path.join(base_dir, subdir)
        if os.path.isdir(subd_fpath):
            shutil.rmtree(subd_fpath, ignore_errors=True)


def interpolate_beat(signal, dim):
    x = np.arange(signal.shape[0])
    f = interpolate.interp1d(x, signal)
    x_new = np.arange(0, dim)
    ls = np.linspace(0, signal.shape[0] - 1, dim)
    y_new = f(ls)
    return x_new, y_new