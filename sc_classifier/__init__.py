""" sc_classifier: a module to run the spatial-color clustering algorithm of Hebert & Macaire. """

from .SCC import SpatialColorClassifier
from .utils import pca_reduce, maxpool, grow, float_rescale, parser
