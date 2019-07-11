""" SCC.py: contains the definition of the SpatialCluster object that performs the
    Hebert & Macaire algorithm """

import numpy as np
from sklearn.cluster import KMeans

class SpatialColorClassifier(object):
    """
    A class to perform the spatial-color pixel clustering algorithm of
    https://ieeexplore.ieee.org/document/4529995 (Hebert & Macaire), based on the
    spectral clustering algorithm, with a pixel similarity matrix based on local color similarity.

    Parameters
    ----------
    data: array-like
        A 2- or 3-D data cube. If 2-d, the data will represent a single color pixel (scalar color).
        If 3-D, the *first* dimension should be the color dimension, e.g. a 200 by 300 RGB image
        should have the shape (3, 200, 300).  data may have an arbitary number of color channels
        but note that we internally construct matrices with n_color dimensions so increasing the
        number of color channels will significantly increase memory usage and computation time.
    n_color_pixels: int
        An integer representing the number of pixels used in the internal pixelization of
        each dimension of color space. 9 works well for most applications, but can be increased
        to discriminate nearby colors (at the expense of performance) or decreased to reduce noise
        (or benefit memory and computation time), if desired. (default: 9)
    n_clusters: int
        The number of clusters (color regions) to segment the map into. Default None means to
        estimate the number of clusters from the image itself. (default: None)
    cluster_eigenvalue_threshold: float
        When estimating the number of clusters from the image itself, a higher
        cluster_eigenvalue_threshold means fewer clusters, and a lower one means more.
        If more or fewer than the desired number of clusters is being found by this algorithm,
        this is the first number to consider changing; see also the maxpool() algorithm in the
        utils module to reduce noise. (In detail, we compute the eigenvalues of the normalized pixel
        adjacency matrix, and estimate the number of clusters as the number of eigenvalues greater
        than this threshold times the maximum eigenvalue.) Must be <=1. (default: 0.9)
    keep_mode_threshold: positive float
        When choosing pixels to group into clusters, we exclude rare pixels. A higher
        keep_mode_threshold excludes more pixels and a lower one excludes fewer (indicating that the
        user wishes to label even fairly unique colors that might be difficult to cluster in color
        space).  In detail, the uniqueness of a given color can be estimated as the sum of the row
        of the adjacency matrix divided by the diagonal value of the adjacency matrix. The smaller
        this number, the more unique the color.  By default, we exclude colors where this ratio is
        less than 1+keep_mode_threshold.  The original paper uses 0.05 although we have found 0 to
        work well for astronomical applications.  (default: 0)
    """
    def __init__(self,
                 data,
                 n_color_pixels=9,
                 n_clusters=None,
                 cluster_eigenvalue_threshold=0.9,
                 keep_mode_threshold=0.0):
        self.data = data
        if n_clusters is not None and n_clusters < 2:
            raise TypeError("Must request at least 2 clusters via n_clusters kwarg")
        if n_clusters is not None and n_clusters != int(n_clusters):
            raise TypeError("n_clusters must be an int")
        self.n_clusters = n_clusters
        if len(self.data.shape) == 2:
            self.data = self.data[None, :, :] # add a spectral dimension
        self.ns = data.shape[0] # number of spectral pixels, ie colors
        # The original paper specifies an odd number, but I don't think that actually matters
        if n_color_pixels != int(n_color_pixels):
            raise TypeError("n_color_pixels must be an integer")
        self.n_color_pixels = n_color_pixels
        self.eigenthresh = cluster_eigenvalue_threshold
        if self.eigenthresh > 1:
            raise TypeError("cluster_eigenvalue_threshold must be <=1")
        if keep_mode_threshold < 0:
            raise TypeError("keep_mode_threshold must be positive")
        self.keep_mode_threshold = keep_mode_threshold
        self.clusters = None

    def get_clusters(self):
        """ Compute the pixel clustering, ie the image segmentation, using the Hebert & Macaire
        algorithm.

        Returns
        -------
        clusters: np.array
            A 2-D map of integer cluster or segment labels. A label of -1 means that no cluster
            could be assigned to the pixel due to the rarity of its color. (See the utils module
            methods grow() and maxpool() for suggestions on how to handle these pixels.)  If
            self.data has shape (n_colors, n_x, n_y), then clusters has shape (n_x, n_y).
        """
        if self.clusters is not None:
            return self.clusters
        pix = self.pixels()
        sccd, weights = self._compute_sccd_weights(pix)
        # if there are no pixels in part of the region, sccd should be 0. We divide by the
        # number of pixels so this may be nan instead; this fixes that problem.
        sccd[weights == 0] = 0

        a, masked_weights, mask = self._get_amatrix(sccd, weights)

        masked_clusters = self._kmeans_clusters(a, masked_weights)

        self.clusters = self._image_clusters(pix, masked_clusters, mask)
        return self.clusters

    def _get_amatrix(self, sccd, weights):
        """ Compute the pixel adjacency matrix, which is the min of sccd in the sub-matrix
        [r1:r2, g1:g2, b1:b2] (or whatever color projection you're using). Since we have an
        arbitrary number of color pixels, this requires some fancy numpy slicing stuff.
        """
        a = np.zeros((self.n_color_pixels**self.ns, self.n_color_pixels**self.ns))
        for i1 in range(self.n_color_pixels**self.ns):
            for i2 in range(self.n_color_pixels**self.ns):
                slices = []
                for ispec in range(self.ns):
                    # how to turn the index of a flat array back to the index of an x**y array
                    c1 = (i1//self.n_color_pixels**(self.ns-ispec-1)) % self.n_color_pixels
                    c2 = (i2//self.n_color_pixels**(self.ns-ispec-1)) % self.n_color_pixels
                    if c2 < c1:
                        # slicing won't work properly and it's a duplicate anyway
                        break
                    slices.append(slice(c1, c2+1))
                else:
                    a[i1, i2] = np.min(sccd[tuple(slices)])
                    a[i2, i1] = np.min(sccd[tuple(slices)])
        flat_weights = weights.flatten()
        mask = (flat_weights != 0) & (np.sum(a, axis=0) > np.diag(a)*(1+self.keep_mode_threshold))
        # we want to mask rows then columns--a[mask, mask] doesn't do that
        a = a[mask]
        a = a[:, mask]

        return a, flat_weights[mask], mask

    def _image_clusters(self, pix, masked_clusters, mask):
        """Take the matrix that defines which color pixels map to which clusters, and use
        the pixelization to turn the data into a map of clusters/image segments. """
        clusters = np.zeros(self.n_color_pixels**self.ns)-1 # -1 to indicate an unclustered pixel
        clusters[mask] = masked_clusters # mask is applied flat
        clusters = clusters.reshape([self.n_color_pixels]*self.ns)

        # Turn [n_colors, nx, ny] into [nx, ny, n_colors],
        rollpix = np.moveaxis(pix, 0, -1)

        _, nx, ny = self.data.shape
        # blank image
        image_clusters = np.zeros((nx, ny))
        for ix in range(nx):
            for iy in range(ny):
                # fill with the corresponding cluster index number
                image_clusters[ix, iy] = clusters[tuple(rollpix[ix, iy])]

        return image_clusters


    def pixels(self):
        """ Return the pixelization for self.data.

        Returns
        -------
        pixels: array_like
            An array with the same shape as self.data, but the values replaced by a pixel in color
            space, in the range [0, self.n_color_pixels).
        """
        pix_list = []
        for color in self.data:
            minc = np.min(color)
            maxc = np.max(color)
            dc = 1.0*(maxc-minc)/self.n_color_pixels
            pixel_vals = (color-minc)/dc
            # this puts maxc in pixel self.n_color_pixels--clip that down
            pixel_vals = np.clip(np.array(pixel_vals, dtype=int), None, self.n_color_pixels-1)
            pix_list.append(pixel_vals)
        pix_list = np.array(np.floor(pix_list), dtype=int)
        return pix_list

    def _compute_sccd_weights(self, pix):
        """ Compute the SCCD, that is, the "spatial-color compactness degree". This has two
        components:
            - HD, the homogeneity degree. Take each spatial pixel and its neighbors *that are
              in the same color pixel as well*, and compute the variance in color space. Average
              over all spatial pixels in that particular color pixel. Compare to the global variance
              for all pixels in that color pixel. This tells you if there's one color grouping in
              that color pixel or several. (E.g., if you had two polka dots that were "hot pink" and
              "light pink" and both were in the "pink" pixel, their local dispersion would be small
              but the global dispersion would be larger because it includes both hot and light pink.
              Making HD lower makes it easier for the clustering algorithm to split these colors.
            - CD, the compactness degree. The average fraction of pixels around each spatial pixel
              that are in the same color pixel. This tells you whether the spatial pixels are
              spatially clustered as a function of color pixel.
        (Technically those are color voxels not color pixels.)  The SCCD is just a multiplication of
        these two quantities.
        """
        ns, nx, ny = self.data.shape
        hd = np.zeros([self.n_color_pixels]*self.ns)
        hd_n = np.zeros([self.n_color_pixels]*self.ns)
        cd = np.zeros([self.n_color_pixels]*self.ns)
        # rollpix is shape (nx, ny, n_color) so you can get the color for each pixel via [ix, iy]
        rollpix = np.moveaxis(pix, 0, -1)
        for ix in range(nx):
            for iy in range(ny):
                # Grab the 3x3 neighbor matrix, make it (3, 3, n_color) in shape
                local_data = self.data[:, max(0, ix-1):min(nx+1, ix+2),
                                       max(0, iy-1):min(ny+1, iy+2)].T.reshape((-1, ns))
                local_pix = rollpix[max(0, ix-1):min(nx+1, ix+2),
                                    max(0, iy-1):min(ny+1, iy+2)
                                   ].reshape((-1, ns))
                var = np.sum(np.var(local_data[local_pix == rollpix[ix, iy]], axis=0))
                hd[tuple(rollpix[ix, iy])] += var
                hd_n[tuple(rollpix[ix, iy])] += 1
                # -1 in next line because this includes the pixel itself
                conn = np.sum(np.all(local_pix == rollpix[ix, iy], axis=1)) - 1
                cd[tuple(rollpix[ix, iy])] += conn

        # More fancy slicing math to allow for an arbitrary number of colors
        global_var = np.zeros(self.n_color_pixels**ns)
        rollpix = rollpix.reshape((-1, ns))
        flat_data = self.data.reshape((-1, ns))
        for i in range(self.n_color_pixels**self.ns):
            loc = [(i//self.n_color_pixels**(self.ns-ispec-1)) % self.n_color_pixels
                   for ispec in range(self.ns)]
            global_var[i] = np.var(flat_data[rollpix == loc])
        global_var = global_var.reshape(hd.shape)

        hd /= hd_n*global_var
        cd /= hd_n
        return cd*hd, hd_n

    def _inv_sqrt_d_matrix(self, a):
        # To normalize the adjacency matrix
        return np.sqrt(np.linalg.inv(np.diag(np.sum(a, axis=0))))

    def _l_matrix(self, a):
        # Normalized adjacency matrix
        inv_sqrt_d = self._inv_sqrt_d_matrix(a)
        return np.dot(np.dot(inv_sqrt_d, a), inv_sqrt_d)

    def _x_matrix(self, a, n):
        # Top n eigenvectors (ordered by eigenvalue) of the normalized adjacency matrix
        l = self._l_matrix(a)
        eigenvalues, eigenvectors = np.linalg.eig(l)
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        return eigenvectors[:, :n]

    def _n_clusters(self, a):
        # Estimate number of clusters by eigenvalue distribution
        if self.n_clusters is not None:
            return self.n_clusters
        l = self._l_matrix(a)
        eigenvalues, _ = np.linalg.eig(l)
        # at least 2 clusters!!
        n_clusters = np.max((np.sum(eigenvalues/np.max(eigenvalues) > self.eigenthresh), 2))
        return n_clusters

    def _y_matrix(self, a, n):
        # A normalized version of the cut-down eigenvector matrix
        x = self._x_matrix(a, n).real # This is okay--all high-level eigenvalues should be real
        inv_sqrt_d = self._inv_sqrt_d_matrix(a)
        return np.dot(inv_sqrt_d, x)

    def _kmeans_clusters(self, a, w):
        # Cluster the eigenvalue representation of the pixels based on the adjacency matrix,
        # weighted by the number of times each pixel appears
        n = self._n_clusters(a)
        # in case you want to plot a or X later
        self.X = self._y_matrix(a, n)
        self.a = a

        # Random state = reproducibility yay!
        kmeans = KMeans(n_clusters=n, random_state=0).fit(self.X, sample_weight=w)
        return kmeans.predict(self.X)
