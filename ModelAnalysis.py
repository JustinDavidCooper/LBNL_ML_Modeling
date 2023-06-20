#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:15:16 2023

@author: jdcooper
"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

import sea_urchin.clustering.metrics as met

import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from analyzer import HyperparamterAnalyzerRFR

from read_structures import GetRawData
from preprocess import trim_data
from preprocess import align_structures
from visuals import show_correlations
from visuals import show_medians
from visuals import show_means

#%% Figure and axis containers

figs= {}
axs= {}

#%% Init

element= "ZnF2"
data= GetRawData(element, rec= False)
data= trim_data(data)

#%% Allign structures and extract Distances

clusters= align_structures( clusters= data['clusters'] )

d0 = met.get_distances(clusters)

#%% Correlation Data

# Spectra
figs['1'], axs['1']= show_correlations(data['nspectras'])

# Structure 
figs['2'], axs['2']= show_correlations(d0)

#%% Data Scaling

scaler= {
    'spectras' : prep.StandardScaler(),
    'd0'      : prep.StandardScaler(),
    # 'd_inv'   : prep.StandardScaler()
    }

data_scaled= {
    'spectras' : scaler['spectras'].fit_transform(data['nspectras']),
    'd0'      : scaler['d0'].fit_transform(d0),
    # 'd_inv'   : scaler['d_inv'].fit_transform(1/d0)
    }

#%% Data Splitting
predictor= 'd0'
regressor= 'spectras'

par_split= {
    'test_size' : 0.2,
    'random_state' : 87
    }

X= {'all' : data_scaled[predictor] }
y= {'all' : data_scaled[regressor] }

X['train'], X['test'], y['train'], y['test'] = train_test_split(X['all'], y['all'], **par_split)

#%% RFR Parameters

par_RFR= {
    'max_features'   :'sqrt',
    # 'oob_score'      : True, #### HyperparameterAnalyzerRFR defaults to use OOB score ####
    # 'max_leaf_nodes' : None,
    'random_state'   : 87
    }

#%% Tree Count Analysis

range_Tree_Counts= np.arange(50,100)


RF_analyzer= HyperparamterAnalyzerRFR(X['all'], y['all'], par_RFR)
history_RF_tc = RF_analyzer.run_tree_scores(range_Tree_Counts)

#%% Plot Tree count convergence

plt.figure()
HyperparamterAnalyzerRFR.plot_summary(history_RF_tc, labels= ['D'])
plt.legend()
plt.xlabel('Tree Count')
plt.ylabel('R^2')

#%% RF Depth

RF_analyzer= HyperparamterAnalyzerRFR(X['all'], y['all'], par_RFR)

range_Depths= np.arange(2,50)

history_RF_depths= RF_analyzer.run_depth_scores(range_Depths)

#%% Plot tree depth convergenct

plt.figure()
HyperparamterAnalyzerRFR.plot_summary(history_RF_depths,labels= ['D'])
plt.legend()
plt.xlabel('Tree Depth')
plt.ylabel('R^2')

#%% RF Model

rf= RandomForestRegressor(n_estimators= history_RF_tc['t_opt'], max_depth= history_RF_depths['d_opt'], **par_RFR)
rf.fit(X['train'],y['train'])

#%% Prediction figures

show_medians(
             erange= data['erange'], 
             prediction= scaler[regressor].inverse_transform(rf.predict(X['test'])), 
             actual= scaler[regressor].inverse_transform(y['test'])
             )

show_means(
           erange= data['erange'],
           prediction= scaler[regressor].inverse_transform(rf.predict(X['test'])),
           actual= scaler[regressor].inverse_transform(y['test'])
           )

#%% Visuals for S2D

# #%% RF Box Plots

# predicted_structures= rf.predict(X['test'])

# fig4, axs4 = plt.subplots(1,y['test'].shape[1])

# for i in range(y['test'].shape[1]):
#     axs4[i].boxplot([predicted_structures[:, i], y['test'][:, i]])  # Create box plot for each feature in A and B
#     axs4[i].set_xticklabels(['Prediction', 'Actual'])  # Set x-axis labels for A and B
#     axs4[i].set_title(f'Feature {i+1}')  # Set title for each subplot

# fig4.suptitle('Random Forest Statistics For {} : CV'.format(element))  # Set the common title
# plt.tight_layout()  # Adjust the layout to avoid overlapping


# #%% RF Example comparison

# MSE_RF = np.mean((y['test'] - predicted_structures)**2,axis= 1)

# barWidth = 0.25
# target= np.argmax(MSE_RF)

# plt.figure()
# plt.bar(np.arange(6), predicted_structures[target,:], color ='peru', width = barWidth,
#         edgecolor ='k', label ='Predicted')
# plt.bar(np.arange(6)+barWidth, y['test'][target,:], color ='burlywood', width = barWidth,
#         edgecolor ='k', label ='Actual')
# plt.title("Worst Average MSE : Sample {}".format(target))
# plt.legend()

# #%% RF MSE Box Plot

# MSE_RF = (y['test'] - predicted_structures)**2

# fig6, axs6 = plt.subplots(1,MSE_RF.shape[1])

# for i in range(MSE_RF.shape[1]):
#     axs6[i].boxplot([MSE_RF[:, i]])  # Create box plot for each feature in A and B
#     axs6[i].set_title(f'Feature {i+1}')  # Set title for each subplot

# fig6.suptitle('RF MSE Statistics For {} : OOB'.format(element))  # Set the common title
# plt.tight_layout()  # Adjust the layout to avoid overlapping




































