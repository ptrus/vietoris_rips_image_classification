import numpy as np
from image_utils import load_all_images,to_grayscale,to_vector
from utils import flatten


def load_dataset(directories=['data/tea_cup', 'data/spoon', 'data/apple']):
    """ Load classes from DIRECTORIES and generate X and Y matrices. """
    classes = map(load_all_images, directories)
    X = flatten([to_vector(to_grayscale(clss)) for clss in classes])
    Y = flatten([[l] * len(clss) for l, clss in enumerate(classes)])
    return np.array(X), np.array(Y)
