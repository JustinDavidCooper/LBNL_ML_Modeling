#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:27:25 2023

@author: jdcooper
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt

import umap

import sea_urchin.alignement.align as ali
import sea_urchin.clustering.metrics as met

from sklearn.ensemble import RandomForestRegressor

#%% Init
from read_structures import GetRawData
from preprocess import trim_data

element= "CaF2"
data= GetRawData(element, rec= True)
data= trim_data(data)

#%% Allign structures and extract Distances

par_alignment = {
    "type"      : "fastoverlap",
    "permute"   : "elements",
    "inversion" : True,
    # "fo_scale"  : 0.8,
    # "fo_maxl"   : 15,
    }

data['clusters'], __ = ali.align_to_mean_structure(
                                                data['clusters'], 
                                                par_alignment,
                                                nmax=10 
                                                )

d0 = met.get_distances(data['clusters'])

#%% UMAP Testing

par_umap = {
    "min_dist" : 0.0,
    "n_components" : 2,
    "n_neighbors" : 15,
    "metric" : 'euclidean',
    'random_state' : 42
    }
par_RFR= {
    'max_features'   :'sqrt',
    'oob_score'      : True,
    'n_estimators'   : 296,
    # 'max_depth'      : None,
    # 'max_leaf_nodes' : None,
    'random_state'   : 87
    }

#%%
range_dimensions = np.arange(2,40)
scores = []
for r in range(12):
    temp_scores= [] 
    par_RFR['random_state']+= 1
    for dim in range_dimensions:
        par_umap['n_components']= dim
        reducer = umap.UMAP(**par_umap)
        UMAP_spec = reducer.fit_transform(data['nspectras'])
        rf= RandomForestRegressor(**par_RFR)
        rf.fit(
            d0,
            UMAP_spec
            )
        temp_scores.append(rf.oob_score_)
    scores= np.array( scores )

#%%
plt.figure()
for i in range(scores.shape[0]):
    plt.scatter(range_dimensions,scores[i,:], color= 'dodgerblue')
plt.xlabel('Dimensions of UMAP Spectra')
plt.ylabel('R^2')
plt.title('RF relation to dimensionality of spectra targets: 95% CI')
    






















