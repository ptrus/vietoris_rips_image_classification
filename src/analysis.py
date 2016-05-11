from dataset import load_dataset
from preprocess import Preprocess
from dionysus import Rips, PairwiseDistances, ExplicitDistances, Filtration
from image_utils import *
from utils import interpolate
import numpy as np
from vietoris_rips import vietoris_rips
from topological_clustering import TopologicalClustering,filter_simplices
from connected_components import connected_components, n_connected_components


def uniq(cx):
    return set(map(tuple, cx))

def critical_edges(dataset, pca_n, skeleton=1):
    """ Return only the edges that connect distinct clusters. """
    n_classes = len(dataset)
    X, y = load_dataset(dataset)
    pp = Preprocess(pca_n)
    X_tr = pp.fit_transform(X)

    distances = PairwiseDistances(X_tr.tolist())
    distances = ExplicitDistances(distances)
    n_samples = len(X_tr)
    indices = range(n_samples)
    old_cx = [[]]
    old_n_components = n_classes
    critical_connections = []
    for r in sorted(set(np.array(distances.distances).flatten())):
        cx = filter_simplices(vietoris_rips(X_tr.tolist(), skeleton, r), skeleton)
        if old_cx != [[]] and old_n_components != n_connected_components((indices, cx)):
            critical_connections.append(list(uniq(cx) - uniq(old_cx))[0])
            # print n_connected_components((indices, cx))
            # print connected_components((indices, cx))
        old_n_components = n_connected_components((indices, cx))
        old_cx = cx
    return critical_connections

def largest_sx(cx):
    return max(cx, key=len)

def all_sxs(dataset, pca_n, skeleton=1):
    """ Return all VR sx-s generated with epsilon such that number of
    clusters = number of classes.
    """
    n_classes = len(dataset)
    X, y = load_dataset(dataset)
    pp = Preprocess(pca_n)
    X_tr = pp.fit_transform(X)

    tc = TopologicalClustering(n_classes)
    tc.fit(X_tr,y)
    return vietoris_rips(X_tr.tolist(), len(X), tc.r2)

def interpolate_edge(dataset, pca_n, edge, ts=0.5, size=(1296, 864)):
    """ Return interpolated set of images from dataset with that lie on the edge. """
    n_classes = len(dataset)
    X, y = load_dataset(dataset)
    pp = Preprocess(pca_n)
    X_tr = pp.fit_transform(X)
    X_inv = pp.inverse_transform(X_tr)
    return [to_image(interpolate(X_inv[edge[0]], X_inv[edge[1]], t / 100.0), size) for t in ts]



dataset = ['../data/apple', '../data/spoon', '../data/apple']
print "edges:", critical_edges(dataset, 0.7)
print "largest dim sx:", largest_sx(all_sxs(dataset, 0.7))
ts = range(0, 100, 20) + [100]
save_all_images(interpolate_edge(dataset, 0.7, critical_edges(dataset, 0.7)[-3], ts=ts), "test")
