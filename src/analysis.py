from dataset import load_dataset
from preprocess import Preprocess
from dionysus import Rips, PairwiseDistances, ExplicitDistances, Filtration
from image_utils import *
from utils import interpolate, flatten
import numpy as np
from vietoris_rips import vietoris_rips
from topological_clustering import TopologicalClustering,filter_simplices
from connected_components import connected_components, n_connected_components
from visualisation import *


def uniq(cx):
    return set(map(tuple, cx))

def invert(Xs):
    global n_classes, X, y, pp, X_tr, X_inv
    Xs_inv = pp.inverse_transform(Xs, only_pca=True)
    return (Xs_inv + 1) / 2.0 * 255


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

def principal_sxs(cx):
    cx_sorted = sorted(cx, key=len, reverse=True)
    vs = map(set, cx_sorted)
    mask = [True] * len(vs)
    for i in range(len(vs)-1):
        if mask[i]:
            v1 = vs[i]
        for j in range(i+1, len(vs)):
            if mask[j]:
                v2 = vs[j]
                if len(v1.intersection(v2)) > 0:
                    mask[j] = False
    return [cx for m,cx in zip(mask, cx_sorted) if m]

def vertex_in_cx(v, cx):
    for sx in cx:
        if v in sx:
            return True
    return False

def sx_mean2(sx, size=(1080, 810)):
    global n_classes, X, y, pp, X_tr, X_inv
    return to_image(invert(np.array([np.mean(X_tr[sx], axis=0)]))[0], size)

def sx_mean(sx, size=(1080, 810)):
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

def interpolate_edge(edge, ts=0.5, size=(1080, 810)):
    """ Return interpolated set of images from dataset with that lie on the edge. """
    global n_classes, X, y, pp, X_tr, X_inv
    if type(ts) is not list:
        ts = [ts]
    return [to_image(invert(interpolate(X_tr[edge[0]], X_tr[edge[1]], t / 100.0)), size) for t in ts]

def get_processed_images(size=(1080, 810)):
    global n_classes, X, y, pp, X_tr, X_inv
    return [to_image(x, size) for x in X_inv]


def prepare_data(dataset, pca_n):
    global n_classes, X, y, pp, X_tr, X_inv
    n_classes = len(dataset)
    X, y = load_dataset(dataset)
    pp = Preprocess(pca_n)
    X_tr = pp.fit_transform(X)
    X_inv = pp.inverse_transform(X_tr)


if __name__ == "__main__":
##    prepare_data(['../data/cup', '../data/paper', '../data/pen'], 0.7)
##    save_all_images(get_processed_images(), "prc")
##    print "edges:", critical_edges()
##    print "largest dim sx:", largest_sx(all_sxs())
##    ts = range(0, 100, 20) + [100]
##    save_all_images(interpolate_edge(critical_edges()[-3], ts=ts), "edge")
##    print "largest dim sx:", save_all_images([sx_mean(largest_sx(all_sxs()))], "sx")
##    print "princpals:", save_all_images([sx_mean(sx) for sx in principal_sxs(all_sxs())], "princ")

# Plot results of single linkage and topological clustering using visualize
    prepare_data(['../data/cup', '../data/paper', '../data/pen'], 0.7)
<<<<<<< HEAD

    #edges
##    print "edges in SL:", critical_edges()
##    print "edges in TC:", filter_simplices(all_sxs(), 1)

    #predictions
    tc = TopologicalClustering(n_classes)
    tc.fit(X_tr,y)
    topo_pred = tc.predict(X_tr)
    print "Topological clustering:   ", topo_pred

    import scipy.spatial.distance as ssd
    from scipy.cluster.hierarchy import linkage, fcluster
    distances = PairwiseDistances(X_tr.tolist())
    distances = ExplicitDistances(distances)
    singlel_pred = fcluster(linkage(ssd.squareform(distances.distances)), n_classes, criterion='maxclust')
    print "Single-linkage clustering:", singlel_pred

    #visualize
    cords = mds_transform(X_tr) #transform points
##    cords_tc = mds_plot(X_tr, topo_pred)
    lines_sl = critical_edges()[:-2] #lines from sl algo
    lines_tc = filter_simplices(all_sxs(), 1) #lines from tc algo
    plot_points(cords, topo_pred) #plot points
    lines_plot(lines_sl, cords, color = "grey")
    plt.show()
    plot_points(cords, topo_pred) #plot points
    lines_plot(lines_tc, cords, color = "grey")
    plt.show()
    

=======
    save_all_images(get_processed_images(), "prc")
    print "edges:", critical_edges()
    print "largest dim sx:", largest_sx(all_sxs())
    ts = range(0, 100, 20) + [100]
    save_all_images(interpolate_edge(critical_edges()[-2], ts=ts), "edge1")
    save_all_images(interpolate_edge(critical_edges()[-1], ts=ts), "edge2")
    print "princpals:", save_all_images([sx_mean2(sx) for sx in principal_sxs(all_sxs())], "princ")
>>>>>>> 51f696a7bf58c312757d88f09e67492af77c1068
