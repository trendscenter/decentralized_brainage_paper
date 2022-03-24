#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.set_printoptions(threshold=sys.maxsize)

"""
=============================================================================
The functions form_XYMatrices*() forms the matrices X, Y.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- inputdir: base directory containing input csv files.
- input_file : csv file containing GM ROI features and brain ages, 
each row for an observation
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - X: Fixed effects design matrix of dimension subjects times features
 - Y: The response matrix of dimension subjects
=============================================================================
"""


def form_XYMatrices(input_dir, input_file):
    X, y, subj_id_list = form_XYMatrices_with_subjects(input_dir, input_file)
    return (X, y)


def form_XYMatrices_with_subjects(input_dir, input_file):
    features = pd.read_csv(os.path.join(input_dir, input_file))
    X = np.array(features[features.columns[:-2]])
    subj_id_list = np.array(features[features.columns[-2:-1]])

    y = features[features.columns[-1:]]
    y = np.array(y[y.columns])[:, 0]
    return (X, y, subj_id_list)


def get_metrics(y_true, y_pred):
    """
    Computes performance evaluation metrics for regression model.
    """
    results = {}
    results['mse'] = mean_squared_error(y_true, y_pred)
    results['rmse'] = math.sqrt(results['mse'])
    results['mae'] = mean_absolute_error(y_true, y_pred)

    return results
