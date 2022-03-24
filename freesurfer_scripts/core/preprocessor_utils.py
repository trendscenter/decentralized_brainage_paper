import os
from random import shuffle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split


def load_npy_data(dir_name, file_name):
    file_with_path = os.path.join(dir_name, file_name)

    with open(file_with_path, "rb") as fp:
        data = np.load(fp)
    return data.astype(np.double)


def split_xy_data(split_type, X, y, test_size, shuffle_data):
    """
    Splits the given datasets into train and test datasets based on split_type.
    """
    if split_type == "age_range_stratified":
        X_train, X_test, y_train, y_test = get_age_range_stratified_split(X, y, test_size, num_bins=4)
    else:
        X_train, X_test, y_train, y_test = get_random_split(X, y, test_size, shuffle_data)

    return X_train, X_test, y_train, y_test


def get_age_range_stratified_split(X, y, test_size, num_bins=4):
    """
    Divide ages into equal size bins and label them
    """
    y_age_range = pd.qcut(y.reshape(y.shape[0]), num_bins, labels=False)

    shufflesplit = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=test_size)
    indx_split = list(shufflesplit.split(X, y_age_range))[0]
    X_train = X[indx_split[0]]
    y_train = y[indx_split[0]]
    X_test = X[indx_split[1]]
    y_test = y[indx_split[1]]

    return X_train, X_test, y_train, y_test


def get_random_split(X, y, test_size, shuffle_data):
    """
    Randomly splits X, y into train and test datasets.

    Args:
        X: Features data
        y: label data
        test_size: ratio of data size to reserve as test set
        shuffle_data: if True shuffles data before splitting.

    Returns: X_train, X_test, y_train, y_test

    """
    return train_test_split(X, y,
                            test_size=test_size,
                            shuffle=shuffle_data)


def get_k_age_range_stratified_partitions(k, X, y, shuffle_data, test_size, num_bins=8):
    k_partitions = []
    """
    Divide ages into equal size bins and labels them
    """
    y_age_range = pd.qcut(y.reshape(y.shape[0]), num_bins, labels=False)

    if k == 1:
        shufflesplit = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=test_size)
        indx_split = list(shufflesplit.split(X, y_age_range))[0]
        k_partitions.append([indx_split[0], indx_split[1]])
    else:
        kfold = StratifiedKFold(n_splits=k, shuffle=shuffle_data, random_state=42)
        for tr_indx, tt_indx in kfold.split(X, y_age_range):
            shufflesplit = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=test_size)
            indx_split = list(shufflesplit.split(X[tt_indx], y_age_range[tt_indx]))[0]
            train_index = tt_indx[indx_split[0]]
            test_index = tt_indx[indx_split[1]]

            k_partitions.append([train_index, test_index])

    return k_partitions


def get_k_age_stratified_partitions(k, X, y, shuffle_data, test_size):
    k_partitions = []

    if k == 1:
        shufflesplit = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=test_size)
        indx_split = list(shufflesplit.split(X, y))[0]
        k_partitions.append([indx_split[0], indx_split[1]])
    else:
        kfold = StratifiedKFold(n_splits=k, shuffle=shuffle_data, random_state=42)
        for tr_indx, tt_indx in kfold.split(X, y):
            """
            Note: Using random split instead of Stratified split. Stratified split does not work as the data in y are
             real values and based on the contents in y it sometimes causes 
            the following error:  ValueError("The least populated class in y has only 1....")
            """
            train_index, test_index = train_test_split(tt_indx, shuffle=True, test_size=test_size)
            k_partitions.append([train_index, test_index])

    return k_partitions


def get_k_random_partitions(k, X, y, shuffle_data, test_size):
    k_partitions = []
    indx_arr = np.arange(len(X))

    if shuffle_data:
        shuffle(indx_arr)

    data_k_splits = np.array_split(indx_arr, k)
    for i in range(len(data_k_splits)):
        spt_indx = data_k_splits[i]
        train_index, test_index = train_test_split(spt_indx, shuffle=True, test_size=test_size)
        k_partitions.append([train_index, test_index])

    return k_partitions


def generate_k_partitions(X, y, subj_ref_data, save_to_dir, k=6, shuffle_data=True, type="age_range_stratified",
                          test_size=0.1,
                          save_partitions=True):
    assert len(X) == len(y) == len(subj_ref_data)

    combined_xy = np.concatenate((X, y), axis=1)
    subj_ref_data_arr = np.array(subj_ref_data)

    if type == "random":
        k_partitions = get_k_random_partitions(k, X, y, shuffle_data, test_size)
    elif type == "age_stratified":
        k_partitions = get_k_age_stratified_partitions(k, X, y, shuffle_data, test_size)
    elif type == "age_range_stratified":
        k_partitions = get_k_age_range_stratified_partitions(k, X, y, shuffle_data, test_size)
    else:
        raise Exception("Please specify the type of split for partitioning.")

    for i in range(len(k_partitions)):
        train_index = k_partitions[i][0]
        test_index = k_partitions[i][1]
        if save_partitions:
            local_dir = save_to_dir + os.sep + f"local{i}" + os.sep + "simulatorRun" + os.sep + type
            os.makedirs(local_dir, exist_ok=True)
            np.savetxt(local_dir + os.sep + f"local{i}_fsv_age_train.csv", combined_xy[train_index], delimiter=",")
            np.savetxt(local_dir + os.sep + f"local{i}_fsv_age_test.csv", combined_xy[test_index], delimiter=",")
            np.savetxt(local_dir + os.sep + f"local{i}_subject_ref_filename_train.csv", subj_ref_data_arr[train_index],
                       fmt='%s',
                       delimiter=",")
            np.savetxt(local_dir + os.sep + f"local{i}_subject_ref_filename_test.csv", subj_ref_data_arr[test_index],
                       fmt='%s',
                       delimiter=",")


def test_functions():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4],
                  [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([0, 1, 2, 3, 1, 1, 2, 3, 2, 0, 1, 3, 0, 3, 2, 0, 1, 3, 0])
    y = np.array(range(len(X)))

    # get_k_random_splits(3, X, y, shuffle_data=True, test_size=0.1)
    # generate_k_splits(X, y, y, "", k=3, shuffle_data=True, type="age_range_stratified")
    get_k_age_range_stratified_partitions(2, X, y, shuffle_data=True, test_size=0.5, num_bins=4)


if __name__ == "__main__":
    test_functions()
