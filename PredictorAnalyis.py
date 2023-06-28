#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:24:43 2023

@author: jdcooper
"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

from preprocess import standard_GetData

import sklearn.preprocessing as prep

from analyzer import HyperparamterAnalyzerRFR
from sklearn.ensemble import RandomForestRegressor

#%% Figure and Axis Containers

figs= {}
axs= {}

#%% Init

par_Data= {
    'element' : 'CaF2',
    'rec' : False,
    'e_cutoff' : None
    }
predictors= [
    'distances',
    'inv_distances',
    'energies'
    # 'forces'
    ]

data= standard_GetData(predictors, **par_Data)

#%% Scaling

scaler= {key:prep.StandardScaler() for key in predictors}
scaler['nspectras']= prep.StandardScaler()

data_scaled= { key:scaler[key].fit_transform(data[key]) for key in predictors }
data_scaled['nspectras']= scaler['nspectras'].fit_transform(data['nspectras'])

#%% RFR Parameters

par_RFR= {
    'max_features'   :'sqrt',
    'oob_score'      : True,
    # 'max_leaf_nodes' : None,
    'random_state'   : 87
    }

#%% Data Splitting

#### For Clarity In Model Fitting ####
X= { key: data[key] for key in predictors }
y= {'all' : data['nspectras'] }

#%% Tree Count analysis

range_Tree_Counts= np.arange(50,100)


analyzers= {key: HyperparamterAnalyzerRFR(X[key], y['all'], par_RFR) for key in predictors}

histories= {key: analyzers[key].run_tree_scores(range_Tree_Counts) for key in predictors}

#%% Plot Tree count convergence

figs['1'], axs['1']= plt.subplots()
HyperparamterAnalyzerRFR.plot_summary(histories)
plt.legend()
plt.xlabel('Tree Count')
plt.ylabel('R^2')

#%% Spectra Cutoff 

range_spectra_end= np.arange(start= 11, stop= data['erange'][-1])
scores= []
for i in range(10):
    temp= []
    for endpoint in range_spectra_end:
        par_RFR['random_state']+= 1
        rf= RandomForestRegressor(**par_RFR)
        rf.fit(
            X= data['distances'],
            y= data['nspectras'][:,data['erange'] <= endpoint]
            )
        temp.append(rf.oob_score_)
    scores.append(temp)
scores= np.array(scores)

#%% Plot Spectra Cutoff convergence

figs['2'], axs['2']= plt.subplots()
for i in range(10):
    plt.scatter(
        range_spectra_end, scores[i,:], color= 'dodgerblue'
        )
plt.plot(range_spectra_end, np.mean(scores, axis= 0), label= 'Mean')

plt.xlabel('Spectra Endpoint')
plt.ylabel('OOB Score')



