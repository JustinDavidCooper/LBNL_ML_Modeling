#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:26:50 2023

@author: jdcooper
"""

#%% Imports

import numpy as np
import sea_urchin.alignement.align as ali

from read_structures import GetRawData
from tools import get_predictor

#%%

def trim_data(data, e_cutoff= None):
    mask = np.all(data['nspectras'] != 0, axis= 0)
    data['erange']= data['erange'][mask]
    data['nspectras']= data['nspectras'][:,mask]
    
    if e_cutoff != None:
        print("Trim")
        data['nspectras']= data['nspectras'][:,data['erange'] <= 15]
        data['erange']= data['erange'][data['erange'] <= 15]
    return data

def align_structures(clusters, nmax= 10, alignment = {
                                                "type"      : "fastoverlap",
                                                "permute"   : "elements",
                                                "inversion" : True
                                                }):


    clusters, __ = ali.align_to_mean_structure(clusters, alignment, nmax= nmax)
    return clusters

def standard_GetData(predictors, element : str, rec= False, e_cutoff= 15):
    data= GetRawData(element, rec= False)
    
    data= trim_data(data, e_cutoff= e_cutoff)
    
    data['clusters']= align_structures( data['clusters'] )
    
    data.update({key: get_predictor(predictor= key, clusters= data['clusters']) for key in predictors})
    
    return data
    