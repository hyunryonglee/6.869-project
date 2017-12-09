import os
import numpy as np
import scipy.misc
import h5py
import struct
import imutils
import cv2

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = ".", data_from = "emnist"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if data_from == "emnist":
        if dataset is "training":
            fname_img = os.path.join(path, 'emnist-balanced-train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'emnist-balanced-train-labels-idx1-ubyte')
        elif dataset is "testing":
            fname_img = os.path.join(path, 'emnist-balanced-test-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'emnist-balanced-test-labels-idx1-ubyte')
        else:
            raise ValueError, "dataset must be 'testing' or 'training'"
    elif data_from == "mnist":
        if dataset is "training":
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        elif dataset is "testing":
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
        else:
            raise ValueError, "dataset must be 'testing' or 'training'"
    else:
        raise ValueError, "data must be from mnist or emnist"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)


np.random.seed(123)

# Loading data from disk
class DataLoader(object):
    def __init__(self, **kwargs):

        self.fine_size = int(kwargs['fine_size'])
        self.data_root = os.path.join(kwargs['data_root'])
        self.type = kwargs['type']
        self.randomize = kwargs['randomize']

        # read data info from lists
        self.training_data = list(read(dataset='training', path=self.data_root, data_from = kwargs['data_from']))
        self.training_data = np.array([np.array(ti, dtype=object) for ti in self.training_data], dtype=object)
        self.testing_data = list(read(dataset='testing', path=self.data_root, data_from = kwargs['data_from']))

        self.num_train = self.training_data.shape[0]
        self.num_test = len(self.testing_data)

        # permute training data
        perm = np.random.permutation(self.num_train)
        self.training_data[:,...] = self.training_data[perm, ...]

        print('# Training Images found:', self.num_train)
        print('# Testing Images found:', self.num_test)

        self._train_idx = 0
        self._test_idx = 0

    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 1))
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            label, pixels = self.training_data[self._train_idx]

            if self.randomize:
                angle = np.random.random_integers(-30,30)
                pixels = imutils.rotate(pixels, angle)

                translate_x = np.random.random_integers(-3,3)
                translate_y = np.random.random_integers(-3,3)
                pixels = imutils.translate(pixels, translate_x, translate_y)

            pixels = pixels.reshape((self.fine_size, self.fine_size, 1))
            images_batch[i, ...] = pixels
            labels_batch[i, ...] = label

            self._train_idx += 1
            if self._train_idx == self.num_train:
                self._train_idx = 0

        return images_batch, labels_batch

    def load_test(self):
        batch_size = len(self.testing_data)
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 1))
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            label, pixels = self.training_data[self._test_idx]
            pixels = pixels.reshape((self.fine_size, self.fine_size, 1))
            images_batch[i, ...] = pixels
            labels_batch[i, ...] = label
            self._test_idx += 1
            if self._test_idx == self.num_test:
                self._test_idx = 0

        return images_batch, labels_batch

    def reset(self):
        self._train_idx = 0
        self._test_idx = 0
