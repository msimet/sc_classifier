""" Utilities to aid in running image segmentation with SpatialCluster: denoising algorithms (grow
and maxpool), test dataset generation, command-line parser, helper function for plotting results."""

import argparse
import numpy as np
import scipy.stats
from sklearn.decomposition import PCA

def maxpool(c):
    """
    Takes an array of integer values "c" and runs a maxpool algorithm, where the maximum value
    in a 3x3 subarray of the image is used to replace the value at the center of that subarray.
    Before the maxpool is implemented, the integer values are reordered so that the highest value
    is the color category with the highest average number of neighbors in the same category,
    which prioritizes dense clusters over broad but less dense noise.

    Parameters
    ----------
    c: array_like
        A 2D array of integers indicating clusters (segments) in an image. Values of -1 are ignored
        for the purposes of the maxpool.

    Returns
    -------
    new_c: array_like
        A new map of clusters (segments) after the cluster reordering and maxpool algorithm. Note
        that the mapping of integer value -> color segment is NOT preserved.
    """
    n = np.max(c)
    nneighbor = np.zeros(int(n)+1)
    npix = np.zeros(int(n)+1)
    nx, ny = c.shape
    for ix in range(nx):
        for iy in range(ny):
            tc = c[ix, iy]
            if tc != -1:
                local_data = c[max(0, ix-1):min(nx+1, ix+2),
                               max(0, iy-1):min(ny+1, iy+2)
                              ].flatten()
                nneighbor[int(tc)] += np.sum(local_data == tc)-1
                npix[int(tc)] += len(local_data)-1
    nneighbor /= npix

    order = np.argsort(nneighbor)

    new_c = np.zeros_like(c)

    for new_i, old_i in enumerate(order):
        new_c[c == old_i] = new_i

    c = new_c
    pool_c = np.zeros_like(c)
    for ix in range(nx):
        for iy in range(ny):
            local_data = c[max(0, ix-1):min(nx+1, ix+2),
                           max(0, iy-1):min(ny+1, iy+2)
                          ]
            pool_c[ix, iy] = np.max(local_data)
    return pool_c

def grow(c):
    """
    Takes an array of integer values "c" and fills in values of -1, indicating no cluster or segment
    assigned, using the mode of the pixel's neighbors. It will proceed iteratively until all values
    are filled, in cases where there are any 3x3 or larger segments of all -1s.

    Parameters
    ----------
    c: array_like
        A 2D array of integers indicating clusters (segments) in an image.

    Returns
    -------
    new_c: array_like
        A new map of clusters (segments) with the values of -1 filled by the closest (dominant)
        pixel values.
    """
    nx, ny = c.shape
    while np.any(c == -1):
        for ix in range(nx):
            for iy in range(ny):
                tc = c[ix, iy]
                if tc == -1:
                    local_data = c[max(0, ix-1):min(nx+1, ix+2),
                                   max(0, iy-1):min(ny+1, iy+2)
                                  ].flatten()
                    local_data = local_data[local_data != -1]
                    if len(local_data) > 0:
                        c[ix, iy] = scipy.stats.mode(local_data, axis=None)[0][0]
    return c

def pca_reduce(X, ndim=3):
    """
    Use the scikit-learn principal component algorithm to reduce the number of color dimensions of a
    multicolor image.  For an array of shape (n_colors, nx, ny), this routine produces a new array
    of shape (ndim, nx, ny).

    Parameters
    ----------
    X: array_like
        A multicolor image. Shape should be (n_colors, nx, ny), that is, the first dimension should
        be the color dimension.
    ndim: int
        The number of desired dimensions in the output array.

    Returns
    -------
    X_new: array_like
        A multicolor image of shape (ndim, nx, ny).
    """
    shape = X.shape
    pca = PCA(n_components=ndim)
    X_flat = X.reshape((shape[0], -1)).T
    X_new = pca.fit_transform(X_flat)
    return X_new.T.reshape((ndim, shape[1], shape[2]))

def test_dataset_1d(seed=0, npix=40, noise_scale=0.03):
    """
    Generate a 1d test dataset, that is, an image with one scalar intensity in each pixel. This
    image looks like a bright central region, a fainter ring around the central region, and then
    a dim outer border.  Noise is applied to the image if noise_scale>0.  The difference in
    intensity between the central region and the ring, and between the ring and the border, is 0.25.

    Parameters
    ----------
    seed: int
        An integer to seed the random number generator (default 0)
    npix: int
        The size of each edge of the image (default 40)
    noise_scale: float
        The width of the Gaussian noise applied to the intensities

    Returns
    -------
    image: array_like
        A 2D array of size (npix, npix)
    """
    np.random.seed(seed)
    x = np.zeros((npix, npix))
    tx = np.arange(npix)
    x[:] = tx - np.mean(tx)
    y = x.T
    rad = np.sqrt(x**2+y**2)
    r = np.zeros((npix, npix))

    r[rad < 0.2*npix] = 1
    r[(rad >= 0.2*npix) & (rad <= 0.4*npix)] = 0.75
    r[rad > 0.4*npix] = 0.5
    if noise_scale > 0:
        r += np.random.normal(scale=noise_scale, size=r.shape)
        r /= np.max(r)
    return r

def test_dataset_3d(seed=0, npix=40, noise_scale=0.03):
    """
    Generate an RGB test dataset. This image looks like a red central region, a green ring around
    the central region, and a blue outer border.  Noise is applied to the image if noise_scale>0.
    The typical difference in intensity between regions within a single color is 0.25.

    Parameters
    ----------
    seed: int
        An integer to seed the random number generator (default 0)
    npix: int
        The size of each edge of the image (default 40)
    noise_scale: float
        The width of the Gaussian noise applied to the intensities

    Returns
    -------
    image: array_like
        A 3D array of size (3, npix, npix)
    """
    np.random.seed(seed)
    x = np.zeros((npix, npix))
    tx = np.arange(npix)
    x[:] = tx - np.mean(tx)
    y = x.T
    rad = np.sqrt(x**2+y**2)
    r = np.zeros((npix, npix))
    g = np.zeros((npix, npix))
    b = np.zeros((npix, npix))

    r[rad < 0.2*npix] = 1
    g[rad < 0.2*npix] = 0.75
    b[rad < 0.2*npix] = 0.75
    r[(rad >= 0.2*npix) & (rad <= 0.4*npix)] = 0.75
    g[(rad >= 0.2*npix) & (rad <= 0.4*npix)] = 1
    b[(rad >= 0.2*npix) & (rad <= 0.4*npix)] = 0.75
    r[rad > 0.4*npix] = 0.75
    g[rad > 0.4*npix] = 0.75
    b[rad > 0.4*npix] = 1
    if noise_scale > 0:
        r += np.random.normal(scale=noise_scale, size=r.shape)
        g += np.random.normal(scale=noise_scale, size=g.shape)
        b += np.random.normal(scale=noise_scale, size=b.shape)
        r /= np.max(r)
        g /= np.max(g)
        b /= np.max(b)
    return np.array([r, g, b])

def float_rescale(x):
    """ A convenience function to apply to arrays being passed to plt.imshow() to ensure the
    colors are scaled correctly. """
    return (1.0*x-np.min(x))/(np.max(x)-np.min(x))

def parser():
    """ Return a command-line parser that includes keys for all the parameters of a SpatialCluster
    object. """
    parser = argparse.ArgumentParser(description='Generate an image segmentation map based on'
                                                 'a spatial-spectral pixqel clustering algorithm.')
    parser.add_argument('-c', '--n_color_pixels', dest='n_color_pixels', type=int, default=9,
                        help='Number of pixels for each dimension of color space (default 9)')
    parser.add_argument('-n', '--n_clusters', dest='n_clusters', type=int,
                        help='Number of image clusters/segments to find (default None, determine '
                             'from image)')
    parser.add_argument('-e', '--cluster_eigenvalue_threshold', dest='cluster_eigenvalue_threshold',
                        type=float, default=0.9,
                        help='Control number of clusters determined from image if n_clusters is'
                             'None; larger number means fewer clusters. Must be <=1. (default 0.9)')
    parser.add_argument('-k', '--keep_mode_threshold', dest='keep_mode_threshold', type=float,
                        default=0,
                        help='Control threshold for trimming outlier color clusters; larger is'
                             'a more aggressive trim. (default 0)')
    return parser
