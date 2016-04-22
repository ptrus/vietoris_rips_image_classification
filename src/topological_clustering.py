# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:29:53 2016

@author: rok
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from dionysus import Rips, PairwiseDistances, Filtration
from vietoris_rips import vietoris_rips
from connected_components import connected_components

class Topo_Cluster(BaseEstimator, TransformerMixin):
    
    """Clustering class.
       Takes points in reduced space (PCA performed on original representation)
       and uses the chosen method to construct clusters.
       
       Args:
       k: number of clusters we want to construct
       method: a topological method used to find clusters
           "VR": using a Vietoris-Ribs complex
           "ALPHA": usin alpha-shapes
    """
    
    def __init__(self, n_clusters, method="VR"):
        assert method in ["VR", "ALPHA"]
        self.n_clusters = n_clusters
        self.method = method
        
    def fit(self, X, skeleton=1, distances=None):
        self.skeleton = skeleton
        self.distances = distances
        if self.method == "VR":
            #Calculate the max cutoff value so that the VR complex has self.k con. comp.
            self.r2 = 10 #Plug in Andrejs function
            #Calculate the min cutoff value so that the VR complex has 1 con. comp.
            self.r1 = 10 #Pluf in Andrejs function
        if self.method == "ALPHA":
            #Calculate some sort of parameters for alpha shapes
            self.r2 = 10
            self.r1 = 10
    
    def predict(self, X):
        simplices = vietoris_rips(X, self.skeleton,  self.r2) #list of lists (points and edges)
        V = range(len(X))
        E = filter(lambda sx: len(sx) == 2, simplices)
        cc = connected_components((V,E))
        return cc
        
        
        