"""
This script aggregates data from all the local sites and performs regression on the combined data.
"""
import json

import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR

from scripts.core import preprocessor_utils as preut
from scripts.core import svr_utils as svrut

TEST_BASE_DIR = "../test/"


def aggregated_SVR(X_train, X_test, y_train, y_test):
    """
    Performs LinearSVR on the passed data.
    """
    input_list = None
    for indx, conf in enumerate(json.loads(open(TEST_BASE_DIR + 'inputspec.json').read())):
        if input_list is None:
            conf.pop("freesurfer_files_path")
            conf.pop("dependents")
            input_list = conf
            break

    regr = make_pipeline(preprocessing.MinMaxScaler(),
                         LinearSVR
                         (epsilon=input_list["epsilon_local"]["value"],
                          tol=input_list["tolerance_local"]["value"],
                          C=input_list["regularization_local"]["value"],
                          loss=input_list["loss_local"]["value"],
                          fit_intercept=input_list["fit_intercept_local"]["value"],
                          intercept_scaling=input_list["intercept_scaling_local"]["value"],
                          dual=input_list["dual_local"]["value"],
                          random_state=input_list["random_state_local"]["value"],
                          max_iter=input_list["max_iterations_local"]["value"]))

    regr.fit(X_train, y_train)
    params = regr.get_params()
    svr2 = params['linearsvr']
    w = svr2.coef_
    w = np.squeeze(w)
    intercept_aggr = svr2.intercept_

    y_train_pred = regr.predict(X_train)
    train_pref = svrut.get_metrics(y_train, y_train_pred)

    y_test_pred = regr.predict(X_test)
    test_pref = svrut.get_metrics(y_test, y_test_pred)

    output_dict = {
        # "intercept_aggregated": intercept_combined.tolist(),
        # "w_aggregated": w.tolist(),
        "n_train_samples_aggregated": len(y_train),
        "n_test_samples_aggregated": len(y_test),
        "rmse_train_aggregated": float(train_pref['rmse']),
        "rmse_test_aggregated": float(test_pref['rmse']),
        "mae_train_aggregated": float(train_pref['mae']),
        "mae_test_aggregated": float(test_pref['mae']),
        "phase": "aggregated",
    }

    # print(output_dict)

    return regr, output_dict


def combine_all_local_data():
    """
    Combines data from all the local sites.
    """
    X_train, X_test, y_train, y_test = get_local_site_data(local_site_num=0)
    inputspec = json.loads(open(TEST_BASE_DIR + 'inputspec.json').read())

    for indx, conf in enumerate(inputspec[1:]):
        [X, y] = svrut.form_XYMatrices(input_dir=TEST_BASE_DIR + f"input/local{indx + 1}/simulatorRun/",
                                       fs_path=conf['freesurfer_files_path']['value'],
                                       dep=conf['dependents']['value'])

        X_train = np.vstack((X_train, X))
        y_train = np.hstack((y_train, y))

    return (X_train, X_test, y_train, y_test)


def combine_all_local_data_sep_test():
    """
    Combines data from all the local sites.
    """
    X_train, y_train, X_test, y_test = None, None, None, None
    for indx, conf in enumerate(json.loads(open(TEST_BASE_DIR + 'inputspec.json').read())):
        [X, y] = svrut.form_XYMatrices(input_dir=TEST_BASE_DIR + f"input/local{indx}/simulatorRun/",
                                       fs_path=conf['freesurfer_files_path']['value'],
                                       dep=conf['dependents']['value'])

        local_X_train, local_X_test, local_y_train, local_y_test = preut.split_xy_data(
            conf['split_type']['value'], X, y, conf['test_size']['value'],
            conf['shuffle']['value'])

        if X_train is None:
            X_train = local_X_train
            y_train = local_y_train
            X_test = local_X_test
            y_test = local_y_test
        else:
            X_train = np.vstack((X_train, local_X_train))
            y_train = np.hstack((y_train, local_y_train))
            X_test = np.vstack((X_test, local_X_test))
            y_test = np.hstack((y_test, local_y_test))

    return (X_train, X_test, y_train, y_test)


def get_local_site_data(local_site_num):
    conf = json.loads(open(TEST_BASE_DIR + 'inputspec.json').read())[local_site_num]
    input_dir = TEST_BASE_DIR + f"input/local{local_site_num}/simulatorRun/"

    [X, y] = svrut.form_XYMatrices(input_dir=input_dir,
                                   fs_path=conf['freesurfer_files_path']['value'],
                                   dep=conf['dependents']['value'])
    local_X_train, local_X_test, local_y_train, local_y_test = preut.split_xy_data(
        conf['split_type']['value'], X, y, conf['test_size']['value'],
        conf['shuffle']['value'])

    return local_X_train, local_X_test, local_y_train, local_y_test


def perform_pca(X_train, X_test):
    """
    Performs PCA for feature reduction
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_pca = scaler.transform(X_train)
    X_test_pca = scaler.transform(X_test)

    pca = PCA(.95)
    pca.fit(X_train_pca)
    X_train_pca = pca.transform(X_train_pca)
    X_test_pca = pca.transform(X_test_pca)

    return X_train_pca, X_test_pca


def build_aggregated_model(pca=False):
    X_train, X_test, y_train, y_test = combine_all_local_data()

    print("Combined train and test data from all the local clients.")
    if pca:
        print("Using PCA features..")
        X_train, X_test = perform_pca(X_train, X_test)

    print("Running SVR now.")
    model, output_dict = aggregated_SVR(X_train, X_test, y_train, y_test)
    print(output_dict)


if __name__ == "__main__":
    build_aggregated_model(pca=False)
