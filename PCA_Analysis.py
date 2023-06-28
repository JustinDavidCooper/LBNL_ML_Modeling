#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:44:27 2023

@author: jcoop
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from umap import UMAP
from sklearn.decomposition import PCA
from statsmodels.api import OLS

from preprocess import standard_GetData

#%% Fig and Ax containers

figs= {}
axs= {}

#%% Init

predictors= [
    'distances'
    # 'inv_distances'
    # 'energies',
    # 'forces'
    ]

data= standard_GetData(
            element= 'CaF2', 
            rec= False, 
            # e_cutoff= 15, 
            predictors= predictors
            )

#%% Parameters

n_components= 2
par_umap= {
        "min_dist" : 0.0,
        "n_components" : n_components,
        "n_neighbors" : 15,
        "metric" : 'euclidean',
        # 'random_state' : 42
        }
par_pca= {
        "n_components" : n_components
        }

#%% Reducers 

umap= UMAP(**par_umap)
pca= PCA(**par_pca)

points = {
    'umap' : umap.fit_transform( data['nspectras'] ),
    'pca'  : pca.fit_transform( data['nspectras'] )
    }

#%% Projections

figs['1'], axs['1'] = plt.subplots(1,2)

# UMAP
axs['1'][0].scatter(points['umap'][:,0], points['umap'][:,1])
axs['1'][0].title.set_text('UMAP Projection')

# PCA
axs['1'][1].scatter(points['pca'][:,0], points['pca'][:,1])
axs['1'][1].title.set_text('PCA Projection')
axs['1'][1].set_xlabel('EVR: {:.2f}'.format(pca.explained_variance_ratio_[0]))
axs['1'][1].set_ylabel('EVR: {:.2f}'.format(pca.explained_variance_ratio_[1]))

#%% Structure Relations

figs['2'], axs['2']= plt.subplots(2,5,sharex= True, sharey= True)

for c in range(2):
    axs['2'][c,0].set_ylabel("Structure Distances")
    axs['2'][c,0].set_xlabel("Spectra PCA EVR: {:.2f}".format(pca.explained_variance_ratio_[c]))
    for d in range(5):
        axs['2'][c,d].scatter(points['pca'][:,c], data['distances'][:,d], s= 10)
        axs['2'][c,d].xaxis.set_ticklabels([])
        axs['2'][c,d].yaxis.set_ticklabels([])

# Title
for d in range(5):
    axs['2'][0,d].title.set_text("d{}".format(d))

figs['3'], axs['3']= plt.subplots(2,5,sharex= True, sharey= True)

for c in range(2):
    axs['3'][c,0].set_ylabel("Structure Distances")
    axs['3'][c,0].set_xlabel("Spectra PCA EVR: {:.2f}".format(pca.explained_variance_ratio_[c]))
    for d in range(5):
        axs['3'][c,d].scatter(points['pca'][:,c], data['distances'][:,5 + d], s = 10)
        axs['3'][c,d].xaxis.set_ticklabels([])
        axs['3'][c,d].yaxis.set_ticklabels([])

# Title
for d in range(5):
    axs['3'][0,d].title.set_text("d{}".format(5 + d))















