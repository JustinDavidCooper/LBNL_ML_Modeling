#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:29:18 2023

@author: jdcooper
"""
#%% Imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from tools import unison_shuffle
from tools import progressBar
import matplotlib.pyplot as plt

#%% Class
class HyperparamterAnalyzerRFR:
    def __init__(self, X, y, parameters= {'max_features' : 'sqrt', 'oob_score' : True }):
        # self.X, self.y = unison_shuffle(X, y, seed= parameters['random_state'])
        self.X= X
        self.y= y
        self.parameters= parameters
    
    def run_depth_scores(self, dRange, CV= None):
        scores= []
        if CV != None:
            print("Cross-Validation not recommended evaluation method for Random Forest Models: Use OOB score")
        p= progressBar(dRange.shape[0])
        for d in dRange:
            rf = RandomForestRegressor(
                                max_depth= d,
                                **self.parameters
                                )
            if self.parameters['oob_score']:
                rf.fit( self.X, self.y )
                scores.append( rf.oob_score_ )
            else:
                scores.append(np.mean(cross_val_score(rf, self.X, self.y, cv= CV)))
            p.tick()
        
        summary= {
            'CV'         : CV,
            'parameters' : self.parameters,
            'depths'     : dRange,
            'scores'     : np.array( scores ),
            'd_opt'      : dRange[ np.argmax(scores) ],
            'score_opt'  : scores[ np.argmax(scores) ]
            }
        return summary
    
    def run_tree_scores(self, tree_sizes, CV= None):
        scores= []
        if CV != None:
            print("Cross-Validation not recommended evaluation method for Random Forest Models: Use OOB score")
        p= progressBar(len(tree_sizes))
        for size in tree_sizes:
            p.tick()
            rf = RandomForestRegressor(
                                n_estimators= size,
                                **self.parameters
                                )
            if self.parameters['oob_score']:
                rf.fit( self.X, self.y )
                scores.append( rf.oob_score_ )
            else:
                scores.append(np.mean(cross_val_score(rf, self.X, self.y, cv= CV)))
        
        summary= {
            'CV'         : CV,
            'parameters' : self.parameters,
            'tree_sizes' : tree_sizes,
            'scores'     : np.array( scores ),
            't_opt'      : tree_sizes[ np.argmax(scores) ],
            'score_opt'  : scores[ np.argmax(scores) ]
            }
        
        return summary

    def plot_summary(summaries:dict):
        optima= []
        for key, summary in summaries.items():
            if 'tree_sizes' in summary:
                plt.plot(summary['tree_sizes'],summary['scores'], label= key)
                optima.append([summary['t_opt'], summary['score_opt']])
            elif 'depths' in summaries[key]:
                plt.plot(summaries[key]['depths'],summaries[key]['scores'], label= key)
                optima.append([summary['d_opt'], summary['score_opt']])
            else:
                raise RuntimeError("{} is not an implemented summarytype".format(key))
        optima= np.array(optima)
        plt.scatter(optima[:,0], optima[:,1], color= 'red', label= 'Max')






