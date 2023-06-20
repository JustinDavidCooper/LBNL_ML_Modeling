#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:59:35 2023

@author: jdcooper
"""
#%% IMPORTS

import numpy as np

#%% DEFS
def unison_shuffle(a,b, seed = 42):
    assert a.shape[0] == b.shape[0]
    perm = np.random.RandomState(seed = seed).permutation(a.shape[0])
    return a[perm,:], b[perm,:]

class progressBar:
    def __init__(self, total, suffix=''):
        self.count_value= 0
        self.total= total
        self.suffix= suffix
        self.show()
        
    def show(self):
        # print(self.count_value, end='')
        bar_length= 100
        filled_up_Length = int(round(bar_length* self.count_value / float(self.total)))
        percentage = round(100.0 * self.count_value/float(self.total),1)
        bar = '█' * filled_up_Length + '-' * (bar_length - filled_up_Length)
        print('\033[F[%s] %s%s ...%s'%(bar, percentage, '%', self.suffix), end= '\r')
        
    def tick(self):
        # input()
        self.count_value += 1
        # print('', end= '\033[F')
        self.show()

def get_coulomb_potential(cluster):
    charge={
        'Ca' : 2,
        'F'  : -1,
        'Zn' : 2,
        'K'  : 1,
        'Li' : 1,
        'Mg' : 2,
        'Na' : 1
        }
    r= cluster.get_all_distances()
    charges= np.array([[charge[symbol] for symbol in cluster.get_chemical_symbols()]])
    energies= charges * charges.T / r
    
    indecies = np.triu_indices(len(cluster), k=1)
    return energies[indecies]

def get_coulomb_potentials(clusters):
    return np.array([get_coulomb_potential(cluster) for cluster in clusters])

#%%

# def test_prog():
#     p = progressBar(10)
#     for i in range(5):
#         p.tick()

# test_prog()