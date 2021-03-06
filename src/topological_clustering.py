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
from visualisation import mds_plot, lines_plot
from matplotlib import pyplot as plt

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

    def __init__(self, n_clusters, method="VR", skeleton=1, plot=False):
        assert method in ["VR", "ALPHA"]
        self.n_clusters = n_clusters
        self.method = method
        self.skeleton = skeleton
        self.plot = plot
        self.cords = []

    def fit(self, X, y=[]):
        ''' y is only used for plotting purposes '''
        if self.plot:
            self.cords = mds_plot(X, y)

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
        if self.plot:
            lines_plot(cx, self.cords)
            plt.show()
        return np.array(connected_components((range(n_samples), filter_simplices(cx, self.skeleton))))


if __name__ == '__main__':
    # from sklearn.datasets import load_iris
    # X = load_iris()['data']
    # y = load_iris()['target']
    # X = X[np.arange(0,100,5),:]
    # y = y[np.arange(0,100,5)]

# Load data
    from sklearn.cross_validation import train_test_split
    from dataset import load_dataset
    datasets = ['../data/cup', '../data/pen', '../data/paper']
    X,y = load_dataset(datasets)
    #X, _, y, _ = train_test_split(X, y, test_size=0.75)
    print "True classes:             ", y

# Preprocess data
    p = Preprocess(0.75)
    X = p.fit_transform(X)

# Apply topological clustering
    n_clusters = len(datasets)
    tc = TopologicalClustering(n_clusters)
    tc.fit(X,y)
    topo_pred = tc.predict(X)
    print "Topological clustering:   ", topo_pred

    import scipy.spatial.distance as ssd
    from scipy.cluster.hierarchy import linkage, fcluster
    distances = PairwiseDistances(X.tolist())
    distances = ExplicitDistances(distances)
    singlel_pred = fcluster(linkage(ssd.squareform(distances.distances)), n_clusters, criterion='maxclust')
    print "Single-linkage clustering:", singlel_pred

    print "Similarity between two predictions:", metrics.adjusted_rand_score(topo_pred, singlel_pred)
