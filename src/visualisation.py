from sklearn import manifold
from matplotlib import pyplot as plt
import numpy as np

def mds_transform(points):
    mds = manifold.MDS()
    coords = mds.fit_transform(points)
    return coords
    
def plot_points(coords, target):
    if target == []:
        target = np.zeros(len(points)).tolist()

    fig = plt.figure()
    ax = plt.axes([0., 0., 1., 1.])
    color_set = 'bgrcmykw'
    colors = [color_set[int(t)] for t in target]
    plt.scatter(coords[:, 0], coords[:, 1], marker='o', s=200, c=colors)

    return coords

def mds_plot(points, target, fig=1):
    if target == []:
        target = np.zeros(len(points)).tolist()
    mds = manifold.MDS()
    coords = mds.fit_transform(points)

    fig = plt.figure(fig)
    ax = plt.axes([0., 0., 1., 1.])
    color_set = 'bgrcmykw'
    colors = [color_set[int(t)] for t in target]
    plt.scatter(coords[:, 0], coords[:, 1], marker='o', s=200, c=colors)

    return coords

def lines_plot(lines, cords, color='r'):
    lines = [line for line in lines if len(line) == 2]
    for line in lines:
        if len(line) != 2: continue
        [p1,p2] = line
        plt.plot([cords[p1, 0], cords[p2, 0]], [cords[p1, 1], cords[p2, 1]], color=color)

if __name__ == '__main__':
    from dataset import load_dataset
    from preprocess import Preprocess, NoScaler
    from sklearn.preprocessing import StandardScaler

    X,Y = load_dataset(['../data/cup', '../data/paper', '../data/pen'])
    cords = mds_plot(X, Y)
    plt.savefig('./mds_no_preprocessing.png')

    p = Preprocess(0.75)#, NoScaler())
    X = p.fit_transform(X)
    cords = mds_plot(X, Y, fig=2)
    plt.savefig('./mds_preprocessing.png')
    #lines = [[1,2],[2,3],[3,1]]
    #lines_plot(lines, cords)
    #plt.show()
