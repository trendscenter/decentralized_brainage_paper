#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script includes the remote computations for brainage prediction using
decentralized SVR with FNC as features
"""
import json
import sys

import numpy as np
from core import common_functions as cf

OUTPUT_KEY_LIST = ['w_local', 'intercept_local', 'n_train_samples_local', 'n_test_samples_local',
                   'rmse_train_local', 'rmse_test_local', 'mae_train_local', 'mae_test_local']

"""
============================================================================
The below function does the following tasks
1. aggregate input parameters from local sites and send it back to local_1
----------------------------------------------------------------------------
This function takes in the following inputs in args['input']:
----------------------------------------------------------------------------
- wlocal : weight coefficients
- intercept_local: intercept of local site
- n_train_samples_local: number of training samples
- n_test_samples_local: number of testing samples
- rmse_train_local: root mean square error
- rmse_test_local: root mean square error of test data
- mae_train_local: mean absolute error
- mae_test_local: mean absolute error of test data
- computation_phase : local_0
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    wlocals : aggregate weight coefficients
    computation_phase : remote_0
- cache:
    intercept_locals: aggregate intercept of local sites
    rmse_train_locals: aggregate root mean square error
    rmse_test_locals: aggregate root mean square error of test data
    mae_train_locals: mean absolute error of training data
    mae_test_locals:  mean absolute error of test data
    n_train_samples_locals: number of training samples
    n_test_samples_locals: number of test samples
============================================================================
"""
def aggregate_locals(input_list, key_list):
    aggegated_dict = {}
    for key_name in key_list:
        aggegated_dict[key_name] = np.array(
            [
                site_dict[key_name]
                for site, site_dict in input_list.items()
                if key_name in site_dict
            ]
        )

    return aggegated_dict


def remote_0(args):
    input_list = args["input"]
    aggegated_dict = aggregate_locals(input_list, OUTPUT_KEY_LIST)

    # dicts
    output_dict = {
        "w_locals": aggegated_dict["w_local"].T.tolist(),
        "phase": "remote_0"
    }

    cache_dict = output_dict.copy()
    cache_dict["intercept_locals"] = aggegated_dict["intercept_local"].tolist()
    cache_dict["rmse_train_locals"] = aggegated_dict["rmse_train_local"].tolist()
    cache_dict["rmse_test_locals"] = aggegated_dict["rmse_test_local"].tolist()
    cache_dict["mae_train_locals"] = aggegated_dict["mae_train_local"].tolist()
    cache_dict["mae_test_locals"] = aggegated_dict["mae_test_local"].tolist()
    cache_dict["n_train_samples_locals"] = aggegated_dict["n_train_samples_local"].tolist()
    cache_dict["n_test_samples_locals"] = aggegated_dict["n_test_samples_local"].tolist()

    result_dict = {"output": output_dict, "cache": cache_dict}

    return json.dumps(result_dict)


"""
============================================================================
The below function does the following tasks
1. organizes the local and owner results for final results output of the
pipeline
----------------------------------------------------------------------------
This function takes in following inputs in args['input'] and args['cache']
----------------------------------------------------------------------------
- input:
    w_owner : weight coefficients of owner site
    intercept_owner: intercept of local site
    rmse_train_owner: root mean square error of training data (owner)
    rmse_test_owner: root mean square error of test data (owner)
    mae_train_owner: mean absolute error of training data (owner)
    mae_test_owner:  mean absolute error of test data (owner)
    n_samples_train_owner: number of samples of training data (owner)
    n_samples_test_owner: number of samples of test data (owner)
    computation_phase : local_1
- cache:
    w_locals : aggregate weight coefficients of local sites
    intercept_locals: aggregate intercept of local sites
    rmse_train_locals: aggregate root mean square error
    rmse_test_locals: aggregate root mean square error of test data
    mae_train_locals: mean absolute error of training data
    mae_test_locals:  mean absolute error of test data
    n_train_samples_locals: number of training samples
    n_test_samples_locals: number of test samples
----------------------------------------------------------------------------
And gives the following output:
----------------------------------------------------------------------------
- output :
    w_owner : weight coefficients of owner site
    intercept_owner: intercept of local site
    rmse_train_owner: root mean square error of training data (owner)
    rmse_test_owner: root mean square error of test data (owner)
    n_samples_train_owner: number of samples of training data (owner)
    n_samples_test_owner: number of samples of test data (owner)
    wlocals : aggregate weight coefficients of local sites
    intercept_locals: aggregate intercept of local sites
    rmse_train_locals: aggregate root mean square error
    rmse_test_locals: aggregate root mean square error of test data
    n_samples_locals: aggregate number of samples
============================================================================
"""
def remote_1(args):
    input_list = args["input"]
    state_list = args["state"]
    owner = state_list["owner"] if "owner" in state_list else "local0"
    dict_owner = input_list[owner]
    dict_locals = args["cache"]

    # combine owner and locals
    output_dict = {
        "w_locals": [dict_locals.get("w_locals"), "arrays"],
        "intercept_locals": [dict_locals.get("intercept_locals"), "array"],

        "w_owner": [dict_owner.get("w_owner"), "array"],
        "intercept_owner": [dict_owner.get("intercept_owner"), "number"],

        "n_train_samples_locals": [dict_locals.get("n_train_samples_locals"), "array"],
        "n_test_samples_locals": [dict_locals.get("n_test_samples_locals"), "array"],
        "rmse_train_locals": [dict_locals.get("rmse_train_locals"), "tables"],
        "rmse_test_locals": [dict_locals.get("rmse_test_locals"), "tables"],
        "mae_train_locals": [dict_locals.get("mae_train_locals"), "tables"],
        "mae_test_locals": [dict_locals.get("mae_test_locals"), "tables"],
        "n_train_samples_owner": [dict_owner.get("n_train_samples_owner"), "number"],
        "n_test_samples_owner": [dict_owner.get("n_test_samples_owner"), "number"],
        "rmse_train_owner": [dict_owner.get("rmse_train_owner"), "table"],
        "rmse_test_owner": [dict_owner.get("rmse_test_owner"), "table"],
        "mae_train_owner": [dict_owner.get("mae_train_owner"), "table"],
        "mae_test_owner": [dict_owner.get("mae_test_owner"), "table"],

    }

    result_dict = {"output": output_dict, "success": True}
    return json.dumps(result_dict)


if __name__ == "__main__":
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(cf.list_recursive(parsed_args, "phase"))

    if "local_0" in phase_key:
        result_dict = remote_0(parsed_args)
        sys.stdout.write(result_dict)
    elif "local_1" in phase_key:
        result_dict = remote_1(parsed_args)
        sys.stdout.write(result_dict)
    else:
        raise Exception("Error occurred at Remote")
