"""
This script aggregates data from all the local sites and performs regression on the combined data.
"""

import json

import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR
from core import preprocessor_utils as preut
from core import svr_utils as svrut


"""
Performs LinearSVR on the passed data.
"""
def aggregated_SVR(X_train, X_test, y_train, y_test):
    input_list = None
    for indx, conf in enumerate(json.loads(open('../test/inputspec.json').read())):
        if input_list is None:
            conf.pop("site_data")
            conf.pop("site_label")
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

    print(output_dict)

    return regr


"""
Combines data from all the local sites.
"""
def combine_all_local_data():
    X_train, y_train, X_test, y_test = None, None, None, None
    for indx, conf in enumerate(json.loads(open('../test/inputspec.json').read())):
        input_dir = f"../test/input/local{indx}/simulatorRun/"
        data_file = conf['site_data']['value'][0]
        label_file = conf['site_label']['value'][0]
        input_source = conf['input_source']['value']
        [X, y] = svrut.form_XYMatrices(input_dir, input_source, data_file, label_file)
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
    conf = json.loads(open('../test/inputspec.json').read())[local_site_num]
    input_dir = f"../test/input/local{local_site_num}/simulatorRun/"
    data_file = conf['site_data']['value'][0]
    label_file = conf['site_label']['value'][0]
    input_source = conf['input_source']['value']
    [X, y] = svrut.form_XYMatrices(input_dir, input_source, data_file, label_file)
    local_X_train, local_X_test, local_y_train, local_y_test = preut.split_xy_data(
        conf['split_type']['value'], X, y, conf['test_size']['value'],
        conf['shuffle']['value'])

    return local_X_train, local_X_test, local_y_train, local_y_test


"""
Performs PCA for feature reduction
"""
def perform_pca(X_train, X_test):
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
    model = aggregated_SVR(X_train, X_test, y_train, y_test)

    """
    Get metrics only for local-0 (holdout) site
    """
    local0_X_train, local0_X_test, local0_y_train, local0_y_test = get_local_site_data(0)
    local0_y_train_pred = model.predict(local0_X_train)
    local0_y_test_pred = model.predict(local0_X_test)
    train_pref = svrut.get_metrics(local0_y_train, local0_y_train_pred)
    test_pref = svrut.get_metrics(local0_y_test, local0_y_test_pred)
    print("Model performance only on local0 train and test data:")
    print("Train data: ", train_pref)
    print("Test data: ", test_pref)

if __name__ == "__main__":
    build_aggregated_model(pca=False)
