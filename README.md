sc_classifier
=============

A package to compute an image segmentation map via the spatial-spectral pixel clustering
algorithm of Hebert & Macaire (https://ieeexplore.ieee.org/document/4529995), developed for use
in astronomical image processing.

The basic algorithm is run via the SpatialColorClassifier object. For example, given a data array
`d` with shape `(n_color_dimensions, n_x, n_y)`, you can get a new image with shape `(n_x, n_y)` and
values `[-1, n_clusters)` via:

```
>>> scc = sc_classifier.SpatialColorClassifier(d)
>>> image_segments = scc.get_clusters()
```

You can add additional keyword arguments that control how the segments are determined.

Additionally, the package contains two algorithms to denoise the resulting maps. `grow()` fills in
pixel values of -1 (unknown segment) with the mode of the surrounding pixels. `maxpool()` runs a 3x3
maxpool algorithm that prioritizes densely clustered segments over more diffuse segments, and is
useful if a large area has been divided into two interlaced color segments, at the expense of
growing or shrinking contiguous regions by a few pixels.

The algorithm relies on an internal matrix that is size `(n_color_pixels)**(n_color_dimensions)`,
where `n_color_pixels` is the number of pixels used in the internal color pixelization scheme.
Therefore, high-dimensional color spaces are demanding in both computation time and memory usage.
`sc_classifier.pca_reduce` applies the `scikit-learn` PCA routine to a data cube to reduce its
color dimensionality.

Two example scripts can be found in the `examples/` directory. `get_clusters_from_file.py` runs the
algorithm on a user-defined FITS file.  `get_clusters_from_array.py` runs the algorithm on a
user-defined data cube. Both can be run with the `--help` option to see available command-line
arguments.
