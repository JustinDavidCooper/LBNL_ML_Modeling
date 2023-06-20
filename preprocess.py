#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:26:50 2023

@author: jdcooper
"""

#%% Imports

import numpy as np
import sea_urchin.alignement.align as ali

#%%

def trim_data(data):
    mask = np.all(data['nspectras'] != 0, axis= 0)
    data['erange']= data['erange'][mask]
    data['nspectras']= data['nspectras'][:,mask]
    return data

def align_structures(clusters, nmax= 10, alignment = {
                                                "type"      : "fastoverlap",
                                                "permute"   : "elements",
                                                "inversion" : True
                                                }):


    clusters, __ = ali.align_to_mean_structure(clusters, alignment, nmax= nmax)
    return clusters