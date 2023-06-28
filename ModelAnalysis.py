#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:15:16 2023

@author: jdcooper
"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

from preprocess import standard_GetData

from visuals import show_correlations

import sklearn.preprocessing as prep
# from sklearn.model_selection import train_test_split

# from sklearn.ensemble import RandomForestRegressor
from analyzer import HyperparamterAnalyzerRFR

# from visuals import show_medians
# from visuals import show_means

#%% Figure and axis containers

figs= {}
axs= {}

#%% Init

predictors= [
    'distances',
    'inv_distances',
    'energies'
    # 'forces'
    ]

data= standard_GetData(
            element= 'CaF2', 
            rec= False, 
            e_cutoff= 15, 
            predictors= predictors
            )

#%% Correlation Data

# Spectra
figs['1'], axs['1']= show_correlations(data['nspectras'])

# Structure 
figs['2'], axs['2']= show_correlations(data['distances'])

#%% Scaling

scaler= {key:prep.StandardScaler() for key in predictors}
scaler['nspectras']= prep.StandardScaler()

data_scaled= { key:scaler[key].fit_transform(data[key]) for key in predictors }
data_scaled['nspectras']= scaler['nspectras'].fit_transform(data['nspectras'])

#%% Data Splitting

X= { key: data[key] for key in predictors }
y= {'all' : data_scaled['nspectras'] }

#%% RFR Parameters

par_RFR= {
    'max_features'   :'sqrt',
    # 'max_leaf_nodes' : None,
    'random_state'   : 87
    }

#%% Tree Count Analysis

range_Tree_Counts= np.arange(50,300)


analyzers= {key: HyperparamterAnalyzerRFR(X[key], y['all'], par_RFR) for key in predictors}

histories_treeCount= {key: analyzers[key].run_tree_scores(range_Tree_Counts) for key in predictors}

#%% Plot Tree count convergence

figs['3'], axs['3'] =plt.subplots()

HyperparamterAnalyzerRFR.plot_summary(histories_treeCount)

plt.legend()
plt.xlabel('Tree Count')
plt.ylabel('R^2')

#%% RF Depth

range_Depths= np.arange(2,25)

histories_depth= {key: analyzers[key].run_depth_scores(range_Depths) for key in predictors}

#%% Plot tree depth convergenct

figs['4'], axs['4']= plt.subplots()

HyperparamterAnalyzerRFR.plot_summary(histories_depth)

plt.legend()
plt.xlabel('Tree Depth')
plt.ylabel('R^2')

# #%% RF Model

# rf= RandomForestRegressor(n_estimators= history_RF_tc['t_opt'], max_depth= history_RF_depths['d_opt'], **par_RFR)
# rf.fit(X['train'],y['train'])

# #%% Prediction figures

# show_medians(
#              erange= data['erange'], 
#              prediction= scaler[regressor].inverse_transform(rf.predict(X['test'])), 
#              actual= scaler[regressor].inverse_transform(y['test'])
#              )

# show_means(
#            erange= data['erange'],
#            prediction= scaler[regressor].inverse_transform(rf.predict(X['test'])),
#            actual= scaler[regressor].inverse_transform(y['test'])
#            )


