import Image
from os import listdir
from utils import *


def load_all_images(directory):
    """ Return a list with all images from DIRECTORY. """
    return [Image.open(directory + '/' + filename) for filename in listdir(directory)]

def save_all_images(images, directory):
    """ Save all IMAGES to the DIRECTORY in png format. """
    for i in range(len(images)):
        images[i].save(directory + '/' + str(i) + '.png')

def resize(images, size):
    """ Resize all IMAGES to SIZE. """
    return do_all(images, lambda i: i.resize(size, Image.ANTIALIAS))


def to_black_and_white(images):
    """ Convert images to black and white. """
    return do_all(images, lambda i: i.convert('LA'))


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
    imgs = to_black_and_white(imgs)
    save_all_images(imgs, save_dir)
