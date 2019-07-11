""" get_clusters.py: a callable Python script to run the Hebert and Macaire clustering algorithm
implemented in the SpatialColorClassifier module. """

import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import CCDData
from sc_classifier import (SpatialColorClassifier as SCC, pca_reduce, maxpool, grow,
                           parser, float_rescale)

def main(filename=None, **kwargs):
    """ Run the clustering algorithm in SpatialColorClassifier; if user requested, make a pretty
        plot comparing the segmentation with the original image; return a map of cluster indices,
        after running a smoothing algorithm if requested by the user.

        For more info, run python get_clusters.py --help"""
    # Prep data and inputs
    if isinstance(filename, str):
        data = np.array(CCDData.read(filename))
    else:
        data = filename
    if kwargs.get('pca', False):
        data = pca_reduce(data, ndim=kwargs.get('n_pca', 3))
    do_maxpool = kwargs.get('maxpool', False)
    do_grow = kwargs.get('grow', False)
    output_file = kwargs.get('output_file', None)
    for k in ['pca', 'n_pca', 'maxpool', 'grow', 'output_file']:
        if k in kwargs:
            del kwargs[k]

    # Perform clustering algorithm
    scc = SCC(data, **kwargs)
    clusters = scc.get_clusters()

    # Make some nice plots
    if output_file is not None:
        fig, axes = plt.subplots(2, 2)
        axes[0][0].imshow(float_rescale(np.rollaxis(data[:3], 0, 3)))
        axes[0][0].set_title("Original image")
        axes[0][1].imshow(clusters)
        axes[0][1].set_title("Original segmentation")

        axes[1][0].imshow(grow(clusters))
        axes[1][0].set_title("Grown segmentation")
        axes[1][1].imshow(maxpool(clusters))
        axes[1][1].set_title("Maxpool segmentation")
        
        fig.tight_layout()

        fig.savefig(output_file)

    if do_grow:
        if do_maxpool:
            return maxpool(grow(clusters))
        return grow(clusters)
    if do_maxpool:
        return maxpool(clusters)
    return clusters

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
    p.add_argument('filename', type=str, help="FITS filename to process")
    return p

if __name__ == '__main__':
    import sys
    p = get_parser()
    kwargs = p.parse_args(sys.argv[1:]).__dict__
    if isinstance(kwargs['filename'], list):
        for f in kwargs['filename']:
            k = kwargs.copy()
            k['filename'] = f
            main(**k)
    else:
        main(**kwargs)
