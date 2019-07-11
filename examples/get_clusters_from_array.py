""" get_clusters_edit_data.py: a callable Python script that runs get_clusters.py, but with
user-defined Python arrays instead of FITS filenames."""

import sys
from sc_classifier import parser
from sc_classifier.utils import test_dataset_3d
from get_clusters_from_file import main

def get_parser():
    """ Add some arguments relating to files & smoothing to the default SpatialColorClassifier
        parser """
    p = parser()
    p.add_argument('-P', '--n_pca', dest='n_pca', type=int, default=3,
                   help="Number of PCA modes to keep")
    p.add_argument('-p', '--pca', dest="pca", action="store_true",
                   help="Run a PCA on the spectral dimension of a data cube")
    p.add_argument('-g', '--grow', dest="grow", action="store_true",
                   help="'Grow' the segmentation map to cover unclustered pixels")
    p.add_argument('-m', '--maxpool', dest="maxpool", action="store_true",
                   help="Run a maxpool algorithm to denoise segmentation map")
    p.add_argument('-o', dest="output_file", type=str,
                   help="Output filename for diagnostic image")
    return p


if __name__ == '__main__':
    # Get a parser and parse command-line arguments
    p = get_parser()
    kwargs = p.parse_args(sys.argv[1:]).__dict__
    # Change the 'filename' arg to be a dataset of your choice
    kwargs['filename'] = test_dataset_3d()
    # run the clustering
    main(**kwargs)
