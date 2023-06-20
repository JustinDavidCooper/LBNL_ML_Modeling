#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:02:58 2023

@author: jdcooper
"""

#%%

import numpy as np
import matplotlib.pyplot as plt

#%%
def show_correlations(a):
    corr_matrix= np.corrcoef(a, rowvar= False)
    fig, ax = plt.subplots()
    im= plt.imshow(corr_matrix, cmap= 'RdBu')
    ax.figure.colorbar(im, ax=ax)
    return fig, ax

def show_medians(erange, prediction, actual, IQR= True):
    plt.figure()
    plt.plot(erange, np.percentile(a= prediction, q=50, axis= 0), color= 'b', label= 'Prediction Median')
    plt.plot(erange, np.percentile(a= actual, q=50, axis= 0), color= 'k', label= 'True Median')
    if IQR:
        plt.fill_between(erange, np.percentile(prediction, q= 25, axis= 0), np.percentile(prediction, q= 75, axis= 0), color= 'b', alpha= 0.2)
        plt.fill_between(erange, np.percentile(actual, q= 25, axis= 0), np.percentile(actual, q= 75, axis= 0), color= 'grey', alpha= 0.2)
    plt.legend()
    plt.title("Medians and IQR" if IQR else "Medians")
    
def show_means(erange, prediction, actual, STD= True):
    mean_pred= np.mean(prediction, axis= 0)
    mean_actual= np.mean(a= actual, axis= 0)
    std_pred = np.std(prediction, axis= 0)
    std_actual= np.std(actual, axis= 0)
    
    plt.figure()
    plt.plot(erange, mean_pred, color= 'blueviolet', label= 'Prediction Mean')
    plt.plot(erange, np.mean(a= actual, axis= 0), color= 'dimgrey', label= 'True Mean')
    if STD:
        plt.fill_between(erange, mean_pred - std_pred, mean_pred + std_pred, color= 'b', alpha= 0.2)
        plt.fill_between(erange, mean_actual - std_actual, mean_actual + std_actual, color= 'grey', alpha= 0.2)
    plt.legend()
    plt.title("Means and STD" if STD else "Means")