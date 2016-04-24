import numpy as np
from matplotlib import pyplot as plt
from dataset import load_dataset
from preprocess import Preprocess
from dionysus import Rips, PairwiseDistances, ExplicitDistances, Filtration
from visualisation import mds_plot, lines_plot
from vietoris_rips import vietoris_rips

def debug(folders, n_components, r = None):
    X,y = load_dataset(folders)
    p = Preprocess(n_components)
    X = p.fit_transform(X)
    
    if r == None:
        distances = PairwiseDistances(X.tolist())
        distances = ExplicitDistances(distances)
        n_samples = len(X)
        r_candidates = sorted(set(np.array(distances.distances).flatten()))
        for r2 in r_candidates:
            print r2
            cx = vietoris_rips(X.tolist(), 1, r2)
            cords = mds_plot(X, y)
            lines_plot(cx, cords)
            plt.show()
    else:
        cx = vietoris_rips(X.tolist(), 1, r)
        cords = mds_plot(X, y)
        lines_plot(cx, cords)
        plt.show()        

if __name__ == '__main__':
    debug(['../train_set/tea_cup', '../train_set/tea_bag'], 0.7)
