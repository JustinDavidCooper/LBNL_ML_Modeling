#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:42:06 2023

@author: jdcooper
"""
#%% Imports

import numpy as np
import glob
import scipy as sp
import umap
import hdbscan
import seaborn as sns
import colorcet

# 
from scipy.stats import pearsonr
import numba

# import sea_urchin.clustering.clusterize as clf
import matplotlib.pyplot as plt
#%% Display Setting
colorset= colorcet.glasbey

scatter_par = {
    's' : 1,
    'marker' : 'o',
    'alpha' : 0.4
    }
#%% Spectra defs
class SpectraCollection:
    def __init__(self, erange_, spectra, labels):
        self.spectra= spectra
        self.erange= erange_
        self.labels= labels
        self.elements = set(labels)
        self.colors= sns.color_palette(colorset, n_colors=len(self.elements)+1)
    
    def get_elem(self,e):
        return self.spectra[self.labels == e]
    
    def plot_spectra(self, e= None, mean= True):
        if e == None:
            if mean:
                plt.plot(self.erange, np.mean(self.spectra, axis= 0).T, color= 'k', label= "All Spectra")
            else:
                plt.plot(self.erange,self.spectra.T, label= "All Spectra")
        else:
            for i, ele in enumerate(e):
                spec= self.get_elem(ele)
                if mean:
                    spec = np.mean(spec, axis= 0)
                    std = np.std(spec, axis= 0)
                    plt.fill_between(self.erange, (spec - std).T, (spec + std).T, color= self.colors[i], alpha= 0.2)
                plt.plot(self.erange,spec.T,color= self.colors[i],label = ele)
                
    def show_spectra(self, e= None, mean= True):
        plt.figure()
        self.plot_spectra(e= e, mean= mean)
        
def read_spectra():
    files = []
    spectras = []
    elements = []
    
    for file in glob.glob("data/*.dat"):
        data = np.loadtxt(file)
        
        spectras.append(data)
        files.append(file)
        elements.append(file.split("_")[1])
    
    spectras = np.array(spectras)
    elements = np.array(elements)
    return files, spectras, elements

def preprocess_spectra(spectras, npoints= 400, decimal= 5, cutoff= np.inf):
    # Narrow range to domain encapsulated by all spectra
    emin = np.max(np.min(spectras[:,:,0], axis= 1))
    emax = np.min(np.max(spectras[:,:,0], axis= 1))
    
    if emax > cutoff:
        emax = cutoff
    
    erange = np.linspace(emin,emax,npoints)
    
    procSpectras = []
    for spectrum in spectras:
        intf = sp.interpolate.interp1d(spectrum[:,0],spectrum[:,1],
                                       fill_value= 0,
                                       bounds_error= False)
        procSpectras.append(intf(erange))
        
    procSpectras= np.round(procSpectras, decimal)
    return erange, procSpectras

#%% Metric Defs
def pearson_coeff(a,b):
    result = pearsonr(a,b)
    return result.statistic

@numba.njit()
def relative_overlap(a,b):
    overlap = np.minimum(a,b)
    A_olp = np.trapz(overlap)
    A_a = np.trapz(a)
    A_b = np.trapz(b)
    return (A_a + A_b) / (5 * A_olp )

from scipy.spatial.distance import mahalanobis    
from sklearn.decomposition import PCA
class PCAMetric:
    def __init__(self,X):
        self.X = X
        self.pca = PCA()
        self.Z = self.pca.fit_transform(X)
        self.ICOV= np.linalg.inv(np.cov(self.Z,rowvar= False))
    
    def PCA_maha(self,a,b):
        x = np.array([a,b])
        z = self.pca.explained_variance_ratio_ * self.pca.transform(x)
        return mahalanobis(z[0,:], z[1,:], self.ICOV)

@numba.njit()
def gaussian( x , s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

def gaussian_blur(x,std=10,coverage=10):
    kernal = np.fromiter( (gaussian( i , std ) for i in range( -coverage, coverage + 1, 1 ) ), float )
    kernal = kernal/np.sum(kernal)
    return np.convolve(x, kernal, mode= 'valid')

def convolution_dist(a,b):
    a_c = gaussian_blur(a)
    b_c = gaussian_blur(b)
    return np.linalg.norm(a_c-b_c)
 
def convolution_integral_dist(a,b):
    a_c = gaussian_blur(a)
    b_c = gaussian_blur(b)
    return relative_overlap(a_c,b_c)

#%% Collection helper defs
class PointCollection:
    def __init__(self,points,labels):
        self.points= points
        self.labels= labels
        self.elements= set(labels)
        self.colors= sns.color_palette(colorset, n_colors=len(self.elements)+1)
        
    def plot_scatter(self, ax= None):
        for i, e in enumerate(self.elements):
            mask = self.labels == e
            if ax == None:
                plt.scatter(
                    x= self.points[mask,0], 
                    y= self.points[mask,1],
                    color = self.colors[i],
                    label = e,
                    **scatter_par
                    )
            else:
                ax.scatter(
                    x= self.points[mask,0], 
                    y= self.points[mask,1],
                    color = self.colors[i],
                    label = e,
                    **scatter_par
                    )
        
class ClusterCollection:
    def __init__(self,spectra,UMAP_points, true_labels, cluster_labels):
        self.spectra= spectra
        self.UMAP_points= UMAP_points
        self.true_labels= true_labels
        self.elements= set(true_labels)
        self.cluster_labels= cluster_labels
        self.clusters = set(cluster_labels)
        self.colors = sns.color_palette(colorset, n_colors=len(self.clusters)+1)
        
        # Make fallout points black
        if -1 in cluster_labels:
            self.colors =  self.colors + [(0,0,0)]
        
    def plot_clusters(self, true_labeling= False):
        ## Color with true labels ##
        if true_labeling:
            colors_true = sns.color_palette(colorset, n_colors=len(self.elements)+1)
            for i, e in enumerate(self.elements):
                mask = self.true_labels == e
                plt.scatter(
                    x= self.UMAP_points[mask,0], 
                    y= self.UMAP_points[mask,1],
                    color = colors_true[i],
                    label = e,
                    **scatter_par
                    ) 
        ## Color with cluster labels ##
        else:           
            for cl in self.clusters:
                mask = self.cluster_labels == cl
                plt.scatter(
                    x= self.UMAP_points[mask,0], 
                    y= self.UMAP_points[mask,1],
                    color = self.colors[cl],
                    label = cl,
                    **scatter_par
                    )
            
    def plot_mean_spectra(self, clust= None, std= False):
        ## Plot all spectra means ##
        if clust == None:
            for cl in self.clusters:
                mask = self.cluster_labels == cl
                spec = np.mean(self.spectra[mask],axis= 0)
                plt.plot(erange,spec.T,c= self.colors[cl], label= cl)
                ## Show stdev range ##
                if std:
                    stdev= np.std(self.spectra[mask], axis= 0)
                    plt.fill_between(
                                    erange,
                                    spec.T - stdev,
                                    spec.T + stdev,
                                    color= self.colors[cl],
                                    alpha= 0.2
                                    )
        ## Plot selected spectra means ##
        else:
            for cl in clust:
                mask = self.cluster_labels == cl
                spec = np.mean(self.spectra[mask],axis=0)
                plt.plot(erange,spec.T,label= cl)
                ## Show stdev range ##
                if std:
                    stdev= np.std(self.spectra[mask], axis= 0)
                    plt.fill_between(
                                    erange,
                                    spec.T - stdev,
                                    spec.T + stdev,
                                    color= self.colors[cl],
                                    alpha= 0.2
                                    )
        
#%% LOAD FILES

#target = "CaF2"
files, spectra, elements = read_spectra()
# Preprocess spectra onto energy domain
erange, proSpectra = preprocess_spectra(spectra, cutoff= 32)

#%% MAIN

spectraC = SpectraCollection(erange, proSpectra,elements)

pca_metric = PCAMetric(spectraC.spectra)

umap_par = {
    "min_dist" : 0.0,
    "n_components" : 2,
    "n_neighbors" : 15,
    "metric" : 'euclidean'
    }
reducer = umap.UMAP(**umap_par)

UMAP_pointC = PointCollection(
                        points= reducer.fit_transform(spectraC.spectra),
                        labels= spectraC.labels
                        )

#%%  HDBSCAN

hdbscan_par = {
    "min_cluster_size" : 15,
    "allow_single_cluster" : True
    }

clusterer = hdbscan.HDBSCAN(
                            **hdbscan_par,
                            )
clusterer.fit(UMAP_pointC.points)
clusterC = ClusterCollection(
                    spectra= spectraC.spectra, 
                    UMAP_points= UMAP_pointC.points, 
                    true_labels= spectraC.labels, 
                    cluster_labels= clusterer.labels_
                    )







