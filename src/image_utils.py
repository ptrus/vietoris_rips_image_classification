import Image
from os import listdir
from utils import do_one_or_list
import numpy as np


def load_all_images(directory):
    """ Return a list with all images from DIRECTORY. """
    return [Image.open(directory + '/' + filename) for filename in listdir(directory)]

def save_all_images(images, directory):
    """ Save all IMAGES to the DIRECTORY in png format. """
    for i in range(len(images)):
        images[i].save(directory + '/' + str(i) + '.png')

def resize(images, size):
    """ Resize all IMAGES to SIZE. """
    return do_one_or_list(images, lambda i: i.resize(size, Image.ANTIALIAS))

def to_grayscale(images):
    """ Convert IMAGES to grayscale. """
    return do_one_or_list(images, lambda i: i.convert('L'))

def to_vector(images):
    """ Convert IMAGES to one-dimensional vectors, with elements of the vector
    being either tuples in case of color image, or just integers for grayscale
    images.
    """
    return do_one_or_list(images, lambda i: np.copy(np.asarray(i.getdata())))

def to_image(vector, size):
    """ Reshape one-dimensional (grayscale) image VECTOR to SIZE and convert
    it to python image object.
    """
    size = (size[1], size[0])
    print size
    print len(vector)
    resized = vector.reshape(size).astype(np.uint8)
    return Image.fromarray(resized)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print "Usage:"
        print " ", sys.argv[0], "[input dir]", "[output dir]", "[size]"
        print "Where:"
        print "  input dir: path to directory containing images,"
        print "  output dir: path to directory that will contain converted images,"
        print "  size: [width]x[height] in px (e.g. 100x100 for 100x100 pixels),"
        sys.exit(1)
    load_dir, save_dir, size = sys.argv[1:]
    w, h = map(int, size.split('x'))
    imgs = load_all_images(load_dir)
    imgs = resize(imgs, (w, h))
    imgs = to_grayscale(imgs)
    save_all_images(imgs, save_dir)
