#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script includes the local computations for brainage prediction using
decentralized SVR with Freesurfer stats features
"""
import json
import os
import sys

import numpy as np
from core import common_functions as cf
from core import preprocessor_utils as preut
from core import svr_utils as svrut
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR

"""
============================================================================
The below function does the following tasks
1. read the dependent variables, fixed covariates from csv files
and forms the X,Y matrices
2. For owner site, just cache the parameters
3. For local sites, preform linear SVR regression to find weight coefficients
and RMSE of training and test data
----------------------------------------------------------------------------
This function takes in the following inputs in args['input']:
----------------------------------------------------------------------------
- freesurfer_files_path : directory path where freesurfer files are stored
- dependents : csv file containing list of brain ages, each row 
for an observation
- arguments for sklearn linear SVR for both owner and local sites
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    wlocal : weight coefficients
    intercept_local: intercept of local site
    rmse_train_local: root mean square error
    rmse_test_local: root mean square error of test data
    n_samples_local: number of samples
    computation_phase : local_0
- cache:
    X_filename : fixed effects design matrix of owner site
    Y_filename : response matrix of owner site
    arguments for sklearn linear SVR for owner
============================================================================
"""
def local_0(args):
    input_list = args['input']
    state_list = args['state']

    input_dir = state_list['baseDirectory']
    cache_dir = state_list["cacheDirectory"]

    owner = state_list["owner"] if "owner" in state_list else "local0"

    fs_path = input_list['freesurfer_files_path']
    dep = input_list['dependents']

    [X, y] = svrut.form_XYMatrices(input_dir, fs_path, dep)

    if state_list["clientId"] == owner:
        np.save(os.path.join(cache_dir, "X.npy"), X)
        np.save(os.path.join(cache_dir, "y.npy"), y)

        cache_dict = {
            "epsilon_owner": input_list["epsilon_owner"],
            "fit_intercept_owner": input_list["fit_intercept_owner"],
            "intercept_scaling_owner": input_list["intercept_scaling_owner"],
            "tolerance_owner": input_list["tolerance_owner"],
            "regularization_owner": input_list["regularization_owner"],
            "loss_owner": input_list["loss_owner"],
            "dual_owner": input_list["dual_owner"],
            "random_state_owner": input_list["random_state_owner"],
            "max_iterations_owner": input_list["max_iterations_owner"],
            "split_type": input_list["split_type"],
            "test_size": input_list["test_size"],
            "shuffle": input_list["shuffle"],
            "X_filename": "X.npy",
            "y_filename": "y.npy",
        }
        output_dict = {"phase": "local_0"}

    else:
        n_samples_local = X.shape[0]
        X = X.astype(np.double)
        regr = make_pipeline(preprocessing.MinMaxScaler(),
                             LinearSVR
                             (epsilon=input_list["epsilon_local"],
                              tol=input_list["tolerance_local"],
                              C=input_list["regularization_local"],
                              loss=input_list["loss_local"],
                              fit_intercept=input_list["fit_intercept_local"],
                              intercept_scaling=input_list["intercept_scaling_local"],
                              dual=input_list["dual_local"],
                              random_state=input_list["random_state_local"],
                              max_iter=input_list["max_iterations_local"]))

        regr.fit(X, y)
        params = regr.get_params()
        svr2 = params['linearsvr']
        w = svr2.coef_
        w = np.squeeze(w)
        intercept_local = svr2.intercept_

        y_pred = regr.predict(X)
        train_perf = svrut.get_metrics(y, y_pred)

        """TODO: Check why is data split after training the model with entire dataset. """
        X_train, X_test, y_train, y_test = preut.split_xy_data(input_list['split_type'], X, y, input_list["test_size"],
                                                               input_list["shuffle"])

        y_test_pred = regr.predict(X_test)
        test_perf = svrut.get_metrics(y_test, y_test_pred)

        cache_dict = {}
        output_dict = {
            "intercept_local": intercept_local.tolist(),
            "w_local": w.tolist(),
            "rmse_train_local": float(train_perf['rmse']),
            "rmse_test_local": float(test_perf['rmse']),
            "mae_train_local": float(train_perf['mae']),
            "mae_test_local": float(test_perf['mae']),
            "n_samples_local": n_samples_local,
            "phase": "local_0",
        }

    result_dict = {"output": output_dict, "cache": cache_dict}
    return json.dumps(result_dict)


"""
============================================================================
The below function does the following tasks
1. read the X,Y matrices for owner site
2. For owner site, 
    a. modify X and Y with weighted values using the mean weight coefficients 
    obtained from local sites
    b. perform linear SVR regression to find weight coefficients
    and RMSE of training and test data
3. For local sites, just output phase value
----------------------------------------------------------------------------
This function takes in following inputs in args['input'] and args['cache']
----------------------------------------------------------------------------
- input:
    wlocals : weight coefficients from all sites
- cache:
    X_filename : fixed effects design matrix of owner site
    Y_filename : response matrix of owner site
    arguments for sklearn linear SVR for owner
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    wlocal : weight coefficients
    intercept_owner: intercept of owner site
    rmse_train_owner: root mean square error of training data (owner)
    rmse_test_owner: root mean square error of test data (owner)
    n_samples_train_owner: number of samples of training data (owner)
    n_samples_test_owner: number of samples of test data (owner)
    computation_phase : local_1
============================================================================
"""
def local_1(args):
    state_list = args["state"]
    owner = state_list["owner"] if "owner" in state_list else "local0"

    if state_list["clientId"] == owner:
        input_list = args["input"]
        cache_list = args["cache"]
        cache_dir = state_list["cacheDirectory"]

        X = preut.load_npy_data(cache_dir, cache_list.get("X_filename", ""))
        y = preut.load_npy_data(cache_dir, cache_list.get("y_filename", ""))

        w_locals = np.array(input_list["w_locals"])
        w_locals = w_locals.astype(np.double)

        w_avg = np.mean(w_locals, axis=1)
        w_avg = w_avg.reshape(-1, 1)

        # split train/test data
        X_train, X_test, y_train, y_test = preut.split_xy_data(cache_list["split_type"], X, y, cache_list["test_size"],
                                                               cache_list["shuffle"])

        U_train = np.matmul(X_train, w_avg)
        U_test = np.matmul(X_test, w_avg)
        n_samples_owner = U_train.shape[0]

        regr = make_pipeline(preprocessing.MinMaxScaler(),
                             LinearSVR(epsilon=cache_list["epsilon_owner"],
                                       tol=cache_list["tolerance_owner"],
                                       C=cache_list["regularization_owner"],
                                       loss=cache_list["loss_owner"],
                                       fit_intercept=cache_list["fit_intercept_owner"],
                                       intercept_scaling=cache_list["intercept_scaling_owner"],
                                       dual=cache_list["dual_owner"],
                                       random_state=cache_list["random_state_owner"],
                                       max_iter=cache_list["max_iterations_owner"]))

        regr.fit(U_train.astype(np.double), y_train)
        params = regr.get_params()
        svr2 = params['linearsvr']
        w_owner = svr2.coef_
        w_owner = np.squeeze(w_owner)
        if cache_list["fit_intercept_owner"]:
            intercept_owner = svr2.intercept_
        else:
            intercept_owner = 0.0

        y_train_pred = regr.predict(U_train)
        y_test_pred = regr.predict(U_test)

        train_perf = svrut.get_metrics(y_train, y_train_pred)
        test_perf = svrut.get_metrics(y_test, y_test_pred)

        output_dict = {
            "w_owner": w_owner.tolist(),
            "intercept_owner": float(intercept_owner),
            "n_samples_train_owner": len(U_train),
            "n_samples_test_owner": len(U_test),
            "rmse_train_owner": float(train_perf['rmse']),
            "rmse_test_owner": float(test_perf['rmse']),
            "mae_train_owner": float(train_perf['mae']),
            "mae_test_owner": float(test_perf['mae']),
            "phase": "local_1"
        }
    else:
        output_dict = {"phase": "local_1"}

    result_dict = {"output": output_dict}
    return json.dumps(result_dict)


if __name__ == "__main__":
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(cf.list_recursive(parsed_args, "phase"))
    if not phase_key:
        result_dict = local_0(parsed_args)
        sys.stdout.write(result_dict)
    elif "remote_0" in phase_key:
        result_dict = local_1(parsed_args)
        sys.stdout.write(result_dict)
    else:
        raise Exception("Error occurred at Local")
