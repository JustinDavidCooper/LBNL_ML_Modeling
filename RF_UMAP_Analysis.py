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
from sklearn.decomposition import PCA

from preprocess import standard_GetData

from sklearn.ensemble import RandomForestRegressor

#%% Fig and Ax containers

figs= {}
axs= {}

#%% Init

predictors= [
    'distances',
    'inv_distances'
    # 'energies',
    # 'forces'
    ]

data= standard_GetData(
            element= 'CaF2', 
            rec= False, 
            e_cutoff= 15, 
            predictors= predictors
            )

#%% UMAP Testing

par_umap = {
    "min_dist" : 0.0,
    "n_components" : 2,
    "n_neighbors" : 15,
    "metric" : 'euclidean',
    # 'random_state' : 42
    }
par_RFR= {
    'max_features'   :'sqrt',
    'oob_score'      : True,
    'n_estimators'   : 227,
    'max_depth'      : 11,
    # 'max_leaf_nodes' : None,
    # 'random_state'   : 87
    }

#%%
range_dimensions = np.arange(2,data['nspectras'].shape[1]-1)

scores_UMAP = []
for r in range(10):
    print('Iteration: {}'.format(r))
    temp_scores= []
    for dim in range_dimensions:
        par_umap['n_components']= dim
        reducer = umap.UMAP(**par_umap)
        UMAP_spec = reducer.fit_transform(data['nspectras'])
        rf= RandomForestRegressor(**par_RFR)
        rf.fit(
            X= data['inv_distances'],
            y= UMAP_spec
            )
        temp_scores.append(rf.oob_score_)
    rf= RandomForestRegressor(**par_RFR)
    rf.fit(
        X= data['inv_distances'],
        y= data['nspectras']
        )
    temp_scores.append(rf.oob_score_)
    scores_UMAP.append(temp_scores)
scores_UMAP= np.array( scores_UMAP )
#%%

scores_PCA= []
range_dimensions = np.arange(2,data['nspectras'].shape[1])

for dim in np.arange(1,data['nspectras'].shape[1]):
    pca = PCA(n_components= dim)
    rf= RandomForestRegressor(**par_RFR)
    rf.fit(
        X= data['inv_distances'],
        y= pca.fit_transform(data['nspectras'])
        )
    scores_PCA.append(rf.oob_score_)
    
scores_PCA= np.array(scores_PCA)

#%%
figs['1'], axs['1']= plt.subplots()

# for i in range(scores_UMAP.shape[0]):
#     plt.scatter(range_dimensions,scores_UMAP[i,:], color= 'dodgerblue', label= 'UMAP')
plt.plot(range_dimensions,np.mean(scores_UMAP,axis=0), color= 'g', label='Mean UMAP')
plt.plot(np.arange(1,data['nspectras'].shape[1]),scores_PCA,color= 'r', label='PCA')
    
plt.xlabel('Dimensions of Spectra')
plt.ylabel('R^2')
plt.legend()
plt.title('Model Accuracy and Dimensionality of Spectra')






















