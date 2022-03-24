#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.set_printoptions(threshold=sys.maxsize)

'''
=============================================================================
The below function forms the matrices X, Y.
-----------------------------------------------------------------------------
It takes as inputs:
-----------------------------------------------------------------------------
- inputdir: base directory containing input csv files.
- fs_path: directory containing freesurfer files subjectwise
- dep : csv file containing list of brain ages, each row for an observation
-----------------------------------------------------------------------------
It returns as outputs:
-----------------------------------------------------------------------------
 - X: Fixed effects design matrix of dimension subjects times features
 - Y: The response matrix of dimension subjects
=============================================================================
'''
def form_XYMatrices(input_dir, fs_path, dep):
    X, y, sub_list = form_XYMatrices_with_subjects(input_dir, fs_path, dep)
    return X, y


def form_XYMatrices_with_subjects(input_dir, fs_path, dep):
    sub_list = os.listdir(os.path.join(input_dir, fs_path))

    X = get_features_from_file(input_dir, fs_path, sub_list)
    y = pd.read_csv(os.path.join(input_dir, dep))
    y = np.array(y[y.columns[0]])

    return X, y, sub_list


def get_features_from_file(input_dir, fs_path, sub_list):
    fs_features_names = ['eTIV_v', 'lh_lateral_ventricle_v', 'lh_cerebellum_cortex_v', 'lh_thalamus_proper_v',
                         'lh_caudate_v', 'lh_putamen_v', 'lh_pallidum_v', 'lh_hippocampus_v', 'lh_amygdala_v',
                         'lh_accumbens_area_v', 'rh_lateral_ventricle_v', 'rh_cerebellum_cortex_v',
                         'rh_thalamus_proper_v', 'rh_caudate_v', 'rh_putamen_v', 'rh_pallidum_v', 'rh_hippocampus_v',
                         'rh_amygdala_v', 'rh_accumbens_area_v', 'lh_bankssts_v', 'lh_caudalanteriorcingulate_v',
                         'lh_caudalmiddlefrontal_v', 'lh_cuneus_v', 'lh_entorhinal_v', 'lh_fusiform_v',
                         'lh_inferiorparietal_v', 'lh_inferiortemporal_v', 'lh_isthmuscingulate_v',
                         'lh_lateraloccipital_v', 'lh_lateralorbitofrontal_v', 'lh_lingual_v',
                         'lh_medialorbitofrontal_v', 'lh_parahippocampal_v', 'lh_parsopercularis_v',
                         'lh_parstriangularis_v', 'lh_pericalcarine_v', 'lh_postcentral_v', 'lh_posteriorcingulate_v',
                         'lh_precentral_v', 'lh_precuneus_v', 'lh_rostralanteriorcingulate_v',
                         'lh_rostralmiddlefrontal_v', 'lh_superiorfrontal_v', 'lh_superiorparietal_v',
                         'lh_superiortemporal_v', 'lh_supramarginal_v', 'lh_temporalpole_v', 'lh_transversetemporal_v',
                         'lh_insula_v', 'rh_bankssts_v', 'rh_caudalanteriorcingulate_v', 'rh_caudalmiddlefrontal_v',
                         'rh_cuneus_v', 'rh_entorhinal_v', 'rh_fusiform_v', 'rh_inferiorparietal_v',
                         'rh_inferiortemporal_v', 'rh_isthmuscingulate_v', 'rh_lateraloccipital_v',
                         'rh_lateralorbitofrontal_v', 'rh_lingual_v', 'rh_medialorbitofrontal_v', 'rh_middletemporal_v',
                         'rh_paracentral_v', 'rh_parsopercularis_v', 'rh_parsorbitalis_v', 'rh_parstriangularis_v',
                         'rh_pericalcarine_v', 'rh_postcentral_v', 'rh_posteriorcingulate_v', 'rh_precentral_v',
                         'rh_precuneus_v', 'rh_rostralanteriorcingulate_v', 'rh_rostralmiddlefrontal_v',
                         'rh_superiorfrontal_v', 'rh_superiorparietal_v', 'rh_superiortemporal_v', 'rh_supramarginal_v',
                         'rh_frontalpole_v', 'rh_temporalpole_v', 'rh_transversetemporal_v', 'rh_insula_v',
                         'lh_bankssts_t', 'lh_caudalanteriorcingulate_t', 'lh_caudalmiddlefrontal_t', 'lh_cuneus_t',
                         'lh_entorhinal_t', 'lh_fusiform_t', 'lh_inferiorparietal_t', 'lh_inferiortemporal_t',
                         'lh_isthmuscingulate_t', 'lh_lateraloccipital_t', 'lh_lateralorbitofrontal_t', 'lh_lingual_t',
                         'lh_medialorbitofrontal_t', 'lh_middletemporal_t', 'lh_parahippocampal_t', 'lh_paracentral_t',
                         'lh_parsopercularis_t', 'lh_parsorbitalis_t', 'lh_parstriangularis_t', 'lh_pericalcarine_t',
                         'lh_postcentral_t', 'lh_posteriorcingulate_t', 'lh_precentral_t', 'lh_precuneus_t',
                         'lh_rostralanteriorcingulate_t', 'lh_rostralmiddlefrontal_t', 'lh_superiorfrontal_t',
                         'lh_superiorparietal_t', 'lh_superiortemporal_t', 'lh_supramarginal_t', 'lh_frontalpole_t',
                         'lh_temporalpole_t', 'lh_transversetemporal_t', 'lh_insula_t', 'lh_MeanThickness_t',
                         'rh_bankssts_t', 'rh_caudalanteriorcingulate_t', 'rh_caudalmiddlefrontal_t', 'rh_cuneus_t',
                         'rh_entorhinal_t', 'rh_fusiform_t', 'rh_inferiorparietal_t', 'rh_inferiortemporal_t',
                         'rh_isthmuscingulate_t', 'rh_lateraloccipital_t', 'rh_lateralorbitofrontal_t', 'rh_lingual_t',
                         'rh_medialorbitofrontal_t', 'rh_middletemporal_t', 'rh_parahippocampal_t', 'rh_paracentral_t',
                         'rh_parsopercularis_t', 'rh_parsorbitalis_t', 'rh_parstriangularis_t', 'rh_pericalcarine_t',
                         'rh_postcentral_t', 'rh_posteriorcingulate_t', 'rh_precentral_t', 'rh_precuneus_t',
                         'rh_rostralanteriorcingulate_t', 'rh_rostralmiddlefrontal_t', 'rh_superiorfrontal_t',
                         'rh_superiorparietal_t', 'rh_superiortemporal_t', 'rh_supramarginal_t', 'rh_frontalpole_t',
                         'rh_temporalpole_t', 'rh_transversetemporal_t', 'rh_insula_t', 'rh_MeanThickness_t',
                         'age_at_cnb'
                         ]
    aseg_features_rownos = [32, 78] + list(range(81, 86)) + [89, 90, 92, 96] + list(range(99, 107))
    lhvol_features_rownos = list(set(list(range(34))) - set([13, 15, 17, 30]))
    rhvol_features_rownos = list(set(list(range(34))) - set([14]))
    lhthick_features_rownos = list(range(35))
    rhthick_features_rownos = list(range(35))
    rownos = [aseg_features_rownos, lhvol_features_rownos, rhvol_features_rownos,
              lhthick_features_rownos, rhthick_features_rownos]

    fs_features_values = []
    for s in sub_list:
        fs_stats_files = ['aseg.stats']
        fs_stats_files.append(s + '_lh_aparc_volume.txt')
        fs_stats_files.append(s + '_rh_aparc_volume.txt')
        fs_stats_files.append(s + '_lh_aparc_thickness.txt')
        fs_stats_files.append(s + '_rh_aparc_thickness.txt')
        feats = []
        for index, f in enumerate(fs_stats_files):
            temp = pd.read_csv(os.path.join(input_dir, fs_path, s, f), delimiter="\t")
            if f.startswith('aseg'):
                feats.append([temp.iloc[rownos[index][0], 0].split()[8].split(',')[0]])
                feats.append([temp.iloc[rownos[index][r], 0].split()[3]
                              for r in range(1, len(rownos[index]))])
            else:
                feats.append((list(temp.iloc[rownos[index], 1])))
        merged = np.array(list(itertools.chain(*feats)), dtype=np.double)
        fs_features_values.append(merged)
    X = np.array(fs_features_values)
    return X


def get_metrics(y_true, y_pred):
    """
    Computes performance evaluation metrics for regression model.
    """
    results = {}
    results['mse'] = mean_squared_error(y_true, y_pred)
    results['rmse'] = math.sqrt(results['mse'])
    results['mae'] = mean_absolute_error(y_true, y_pred)

    return results
