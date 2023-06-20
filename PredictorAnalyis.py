#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:24:43 2023

@author: jdcooper
"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

from read_structures import GetRawData
from preprocess import trim_data

from preprocess import align_structures

from tools import get_coulomb_potentials
import sea_urchin.clustering.metrics as met

import sklearn.preprocessing as prep

from analyzer import HyperparamterAnalyzerRFR


#%%

element= "CaF2"
data= GetRawData(element, rec= False)

data= trim_data(data)

#%% Allign structures

clusters = align_structures(clusters= data['clusters'])

#%% Extract Coulomb Potentials and Distances

data['energies']= get_coulomb_potentials(clusters)
data['distances']= met.get_distances(clusters)
data['inv_distances']= 1/data['distances']

#%% Scaling

scaler= {
    'spectras' : prep.StandardScaler(),
    'distances'      : prep.StandardScaler(),
    'inv_distances'   : prep.StandardScaler(),
    'energies' :prep.StandardScaler()
    }

data_scaled= {
    'spectras' : scaler['spectras'].fit_transform(data['nspectras']),
    'distances'      : scaler['distances'].fit_transform(data['distances']),
    'inv_distances'   : scaler['inv_distances'].fit_transform(1/data['distances']),
    'energies' : scaler['energies'].fit_transform(data['energies'])
    }

#%% RFR Parameters

par_RFR= {
    'max_features'   :'sqrt',
    # 'oob_score'      : True, #### HyperparameterAnalyzerRFR defaults to use OOB score ####
    # 'max_leaf_nodes' : None,
    'random_state'   : 87
    }

#%% Data Splitting

#### For Clarity In Model Fitting ####
X= {'distances'     : data['distances'],
    'inv_distances' : data['inv_distances'],
    'energies'      : data['energies']
    }
y= {'all' : data_scaled['spectras'] }

#%% Tree Count analysis

range_Tree_Counts= np.arange(50,200)


RF_analyzer_d= HyperparamterAnalyzerRFR(X['distances'], y['all'], par_RFR)
RF_analyzer_invd= HyperparamterAnalyzerRFR(X['inv_distances'], y['all'], par_RFR)
RF_analyzer_e= HyperparamterAnalyzerRFR(X['energies'], y['all'], par_RFR)

history_RF_tc_d= RF_analyzer_d.run_tree_scores(range_Tree_Counts)
history_RF_tc_invd= RF_analyzer_invd.run_tree_scores(range_Tree_Counts)
history_RF_tc_e= RF_analyzer_e.run_tree_scores(range_Tree_Counts)

#%% Plot Tree count convergence

plt.figure()
HyperparamterAnalyzerRFR.plot_summary(history_RF_tc_d, history_RF_tc_invd, history_RF_tc_e, labels= ['Distances', 'Inv_distances', 'Energies'])
plt.legend()
plt.xlabel('Tree Count')
plt.ylabel('R^2')










