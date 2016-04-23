# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:29:53 2016

@author: rok
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from dionysus import Rips, PairwiseDistances, ExplicitDistances, Filtration
from vietoris_rips import vietoris_rips
from connected_components import connected_components, n_connected_components
from preprocess import Preprocess
from utils import linear_search
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

def filter_simplices(cx, dim):
    """ Only keep simplices in complex CX of dimension DIM. """
    return filter(lambda sx: len(sx) == dim+1, cx)

class TopologicalClustering(BaseEstimator, TransformerMixin):
    
    """Clustering class.
       Takes points in reduced space (PCA performed on original representation)
       and uses the chosen method to construct clusters.
       
       Args:
       k: number of clusters we want to construct
       method: a topological method used to find clusters
           "VR": using a Vietoris-Ribs complex
           "ALPHA": usin alpha-shapes
    """
    
    def __init__(self, n_clusters, method="VR", skeleton=1):
        assert method in ["VR", "ALPHA"]
        self.n_clusters = n_clusters
        self.method = method
        self.skeleton = skeleton
        
    def fit(self, X):
        distances = PairwiseDistances(X.tolist())
        distances = ExplicitDistances(distances)
        n_samples = len(X)
        r_candidates = sorted(set(np.array(distances.distances).flatten()))
        if self.method == "VR":
            n_cc_for_vr =  lambda r: n_connected_components((range(n_samples), filter_simplices(vietoris_rips(X.tolist(), self.skeleton, r), self.skeleton)))
            self.r1 = linear_search(1, r_candidates, n_cc_for_vr)
            self.r2 = linear_search(self.n_clusters, reversed(r_candidates), n_cc_for_vr)
        if self.method == "ALPHA":
            raise Exception('support for alpha shapes not yet implemented')
    
    def predict(self, X):
        cx = vietoris_rips(X.tolist(), self.skeleton, self.r2)
        n_samples = len(X)
        print cx
        return connected_components((range(n_samples), filter_simplices(cx, self.skeleton)))

if __name__ == '__main__':
    # from sklearn.datasets import load_iris
    # from sklearn.cross_validation import train_test_split
    # X = load_iris()['data']
    # y = load_iris()['target']
    # X, _, y, _ = train_test_split(X, y, test_size=0.75)

    from dataset import load_dataset
    #X,Y = load_dataset(['../data/tea_cup', '../data/spoon', '../data/apple'])
    X,Y = load_dataset(['../train_set/tea_cup', '../train_set/tea_bag'])

    p = Preprocess(0.95)
    X = p.fit_transform(X)
    print len(X[0])

    tc = TopologicalClustering(2)
    tc.fit(X)
    pred = tc.predict(X)
    print np.array(tc.predict(X))
    print "Score:", metrics.adjusted_rand_score(Y, pred)
    ac = AgglomerativeClustering(2)
    sklearn_pred = ac.fit_predict(X)
    print "Benchmark score:", metrics.adjusted_rand_score(Y, sklearn_pred)
    # print y
