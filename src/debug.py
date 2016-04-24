import numpy as np
from matplotlib import pyplot as plt
from dataset import load_dataset
from preprocess import Preprocess
from dionysus import Rips, PairwiseDistances, ExplicitDistances, Filtration
from visualisation import mds_plot, lines_plot
from vietoris_rips import vietoris_rips
from topological_clustering import filter_simplices
from itertools import combinations

def debug(folders, n_components, r = None, max_dimension = 1):
    X,y = load_dataset(folders)
    p = Preprocess(n_components)
    X = p.fit_transform(X)
    
    if r is None:
        distances = PairwiseDistances(X.tolist())
        distances = ExplicitDistances(distances)
        n_samples = len(X)
        r_candidates = sorted(set(np.array(distances.distances).flatten()))
        for r2 in r_candidates:
            print r2
            cx = vietoris_rips(X.tolist(), max_dimension, r2)
            cords = mds_plot(X, y)
            lines_plot(cx, cords)
            plt.show()
    else:
        cx = vietoris_rips(X.tolist(), max_dimension, r)
        actual_max_dimension = len(max(cx, key=len)) - 1
        for d in range(actual_max_dimension, 2, -1):
            sx_d = filter_simplices(cx, d)
            print "dimension", d, ":", len(sx_d), "simplices"
            for i, sx in enumerate(sx_d):
                print i, "..."
                cords = mds_plot(X, y)
                edges = list(combinations(sx, 2))
                lines_plot(edges, cords, color=np.random.rand(3,))
                plt.show()

if __name__ == '__main__':
    debug(['../train_set/tea_cup', '../train_set/tea_bag'], 0.7, r=400, max_dimension=40)
