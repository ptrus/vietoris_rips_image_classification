from dionysus import Rips, PairwiseDistances, Filtration

from preprocess import Preprocess

def vietoris_rips(points, skeleton, max, distances = None):
    """
    Generate the Vietoris-Rips complex on the given set of points in 2D.
    Only simplexes up to dimension skeleton are computed.
    The max parameter denotes the distance cut-off value.
    The distances parameter can be used to precompute distances.
    """

    if distances is None:
        distances = PairwiseDistances(points)

    rips = Rips(distances)

    simplices = Filtration()
    rips.generate(skeleton, max, simplices.append)
    return [list(simplex.vertices) for simplex in simplices]

if __name__ == '__main__':
    from dataset import load_dataset
    x,y = load_dataset(['../data/tea_cup', '../data/spoon', '../data/apple'])

    p = Preprocess(0.7)
    x = p.fit_transform(x)

    # vietoris_rips expect a python list.
    x = x.tolist()
    s = vietoris_rips(x, 4, 500)
    print s
