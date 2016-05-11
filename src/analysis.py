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

def critical_edges(skeleton=1):
    """ Return only the edges that connect distinct clusters. """
    global n_classes, X, y, pp, X_tr, X_inv

    distances = ExplicitDistances(PairwiseDistances(X_tr.tolist()))
    n_samples = len(X_tr)
    indices = range(n_samples)
    old_cx = [[]]
    old_n_components = n_classes
    edges = []
    for r in sorted(set(np.array(distances.distances).flatten())):
        cx = filter_simplices(vietoris_rips(X_tr.tolist(), skeleton, r), skeleton)
        if old_cx != [[]] and old_n_components != n_connected_components((indices, cx)):
            edges.append(list(uniq(cx) - uniq(old_cx))[0])
            # print n_connected_components((indices, cx))
            # print connected_components((indices, cx))
        old_n_components = n_connected_components((indices, cx))
        old_cx = cx
    return edges

def largest_sx(cx):
    return max(cx, key=len)

def sx_mean(sx, size=(1296, 864)):
    global n_classes, X, y, pp, X_tr, X_inv
    return to_image(np.mean(X_inv[sx], axis=0), size)

def all_sxs(skeleton=1):
    """ Return all VR sx-s generated with epsilon such that number of
    clusters = number of classes.
    """
    global n_classes, X, y, pp, X_tr, X_inv
    tc = TopologicalClustering(n_classes)
    tc.fit(X_tr,y)
    return vietoris_rips(X_tr.tolist(), len(X), tc.r2)

def interpolate_edge(edge, ts=0.5, size=(1296, 864)):
    """ Return interpolated set of images from dataset with that lie on the edge. """
    global n_classes, X, y, pp, X_tr, X_inv
    if type(ts) is not list:
        ts = [ts]
    return [to_image(interpolate(X_inv[edge[0]], X_inv[edge[1]], t / 100.0), size) for t in ts]



def prepare_data(dataset, pca_n):
    global n_classes, X, y, pp, X_tr, X_inv
    n_classes = len(dataset)
    X, y = load_dataset(dataset)
    pp = Preprocess(pca_n)
    X_tr = pp.fit_transform(X)
    X_inv = pp.inverse_transform(X_tr)


if __name__ == "__main__":
    prepare_data(['../data/tea_bag', '../data/spoon', '../data/tea_cup'], 0.7)

    print "edges:", critical_edges()
    print "largest dim sx:", largest_sx(all_sxs())
    ts = range(0, 100, 20) + [100]
    save_all_images(interpolate_edge(critical_edges()[-3], ts=ts), "edge")
    print "largest dim sx:", save_all_images([sx_mean(largest_sx(all_sxs()))], "sx")
