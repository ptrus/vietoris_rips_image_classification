import numpy as np
from image_utils import load_all_images,to_grayscale,to_vector
from utils import *


def load_dataset(directories=['data/tea_cup', 'data/spoon', 'data/apple']):
    """ Load classes from DIRECTORIES and generate X and Y matrices. """
    images = map(load_all_images, directories)
    Y = flatten([[d] * len(clss) for d, clss in zip(directories, images)])
    X = flatten([to_vector(to_grayscale(clss)) for clss in images])
    return np.array(X), np.array(Y)
