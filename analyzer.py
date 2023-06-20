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
    def __init__(self, X, y, parameters= {'max_features' : 'sqrt', 'oob_score' : True, 'random_state' : 0 }):
        self.X, self.y = unison_shuffle(X, y, seed= parameters['random_state'])
        self.parameters= parameters
    
    def run_depth_scores(self, dRange, CV= None):
        scores= []
        self.parameters['oob_score']=  CV == None
        # print('Running Depths:')
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
        self.parameters['oob_score']=  CV == None 
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

    def plot_summary(*summaries, labels):
        for i, summary in enumerate(summaries):
            if 'tree_sizes' in summary:
                plt.plot(summary['tree_sizes'],summary['scores'], label= labels[i])
            else:
                plt.plot(summary['depths'],summary['scores'], label= labels[i])
        
        plt.scatter([summary['t_opt'] if 't_opt' in summary else summary['d_opt'] for summary in summaries], [summary['score_opt'] for summary in summaries], color= 'red', label= 'Max')






