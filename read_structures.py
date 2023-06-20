#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:08:12 2023

@author: roncoroni
"""
import glob

#import ase
import ase.io
import numpy as np
import scipy as sp


#%%

def make_cluster(structure, cc, bonds, rec=False):
    # translate to origin
    new_pos = structure.get_distances(cc, range(len(structure)),
                                vector=True, mic=True)
    structure.set_positions(new_pos)

    indexes = np.concatenate(([cc], bonds[cc]))

    if rec:
        for bond in bonds[cc]:
            for atom in bonds[bond]:
                if structure[atom].symbol == structure[bond].symbol:
                    continue
                indexes = np.concatenate((indexes, [atom]))

    indexes = np.unique(indexes)


    cluster = structure[indexes]

    # translate to origin
    origin = structure.get_positions()[cc]
    cluster.translate(-origin)

    cluster.set_pbc(False)
    cluster.set_cell(None)

    return cluster

# def plot_spectra_from_umap(spectras, erange, clusterer):

#     labels = clusterer.labels_

#     colors = sns.color_palette(colorcet.glasbey, n_colors=np.max(labels)+1)
#     if -1 in labels:
#         colors = colors + [(0,0,0)]

#     plt.figure()

#     for label in set(labels):

#         if label == -1:
#             continue

#         tspectras = spectras[labels==label]

#         average_line = np.mean(tspectras, axis=0)
#         std_dev = np.std(tspectras, axis=0)

#         # Plot the average line
#         plt.plot(erange, average_line, color=colors[label], lw=2)
#                  # label=list(set(elements))[label])

#         # Plot the standard deviation band
#         plt.fill_between(
#             erange, np.maximum(average_line - std_dev,0), average_line + std_dev,
#             color=colors[label], alpha=0.4)

#     plt.xlabel("Energy [eV]")
#     plt.ylabel("Intensity [-]")

#     plt.tight_layout()
#     # plt.legend()


#     return

#%%
from ase import neighborlist

def GetRawData(element, rec= False):
    emin    = 6#np.min(spectras[:,:,0])
    emax    = 22#30#np.max(spectras[:,:,0])
    npoints = 300
    
    erange = np.linspace(emin, emax, npoints)
    
    
    
    # element = "CaF2"
    # element = "MgF2"
    # element = "KF"
    #element = "ZnF2"
    
    nspectras = []
    files     = []
    spectras  = []
    elements  = []
    clusters  = []
    for file in glob.glob("data/spectra_{}_*.dat".format(element)):
    
        string = file.split("_")
    
        element = string[1]
        step    = "".join([str(s) for s in string[2] if s.isdigit()])
        nF      = int("".join([str(s) for s in string[3] if s.isdigit()]))-1
    
        # read spectra
        data = np.loadtxt(file)
    
        spectras.append(data)
    
        intf = sp.interpolate.interp1d(data[:,0], data[:,1],
                                       fill_value   = 0,
                                       bounds_error = False)
        nspectras.append(intf(erange))
    
        filein = "structures/qe_{}_{}.in".format(element, step)
    
        structure = ase.io.read(filein)
    
        cutOff = np.array(neighborlist.natural_cutoffs(structure))*1.02
    
        # calculate neighbor list
        neighborList = neighborlist.NeighborList(cutOff,
                                                 self_interaction=False,
                                                 bothways=True)
        neighborList.update(structure)
    
        bonds = neighborList.nl.neighbors
    
        clusters.append(make_cluster(structure, nF, bonds, rec= rec))
    
    
    
        files.append(file)
        elements.append(element)
    
    
    spectras = np.array(spectras)
    nspectras = np.array(nspectras)
    elements = np.array(elements)
    data= {
        "erange" : erange,
        "spectra" : spectras,
        "nspectras" : nspectras,
        "elements" : elements,
        "clusters" : clusters
        }
    return data

#%% UMAP hyperparameter analysis

# fig, axs= plt.subplots(nrows=3,ncols=3,sharex= True,sharey= True)
# mds= [0.0,0.5,1]
# n_ns= [15,50,75]
# PCs= []
# for row, md in enumerate(mds):
#     for col, n in enumerate(n_ns):
#         umap_pars["min_dist"]= md
#         umap_pars["n_neighbors"]= n        
#         reducer= umap.UMAP(**umap_pars)
#         PCs.append(PointCollection(
#                             reducer.fit_transform(data), 
#                             elements
#                             ))
#         PCs[-1].plot_scatter(ax= axs[row,col])
# fig.text(0.06, 0.5, 'min_dist: {}, {}, {}'.format(mds[2],mds[1],mds[0]), ha='center', va='center', rotation='vertical')
# fig.text(0.5, 0.04, 'n_neighbors: {}, {}, {}'.format(n_ns[0],n_ns[1],n_ns[2]), ha='center', va='center')


#%%

# # plot_spectra_from_umap(nspectras, erange, clm_spec)

# # umap = pipes["umap"]
# coors = np.linspace((-1,6), (10,10), 10)
# plt.figure()

# # inv_S1 = pipes.inverse_transform([[6,13]])
# inv_S2 = pipe_spec.inverse_transform(coors)
# #inv_S2 = scaler.inverse_transform(inv_S2)
# # plt.plot(erange, inv_S1[0])
# plt.plot(erange, np.mean(nspectras, axis=0), c="k", lw=3)

# plt.plot(erange, inv_S2.T)



