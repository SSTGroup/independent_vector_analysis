#     Copyright (c) <2021> <University of Paderborn>
#     Signal and System Theory Group, Univ. of Paderborn, https://sst-group.org/
#     https://github.com/SSTGroup/independent_vector_analysis
#
#     Permission is hereby granted, free of charge, to any person
#     obtaining a copy of this software and associated documentation
#     files (the "Software"), to deal in the Software without restriction,
#     including without limitation the rights to use, copy, modify and
#     merge the Software, subject to the following conditions:
#
#     1.) The Software is used for non-commercial research and
#        education purposes.
#
#     2.) The above copyright notice and this permission notice shall be
#        included in all copies or substantial portions of the Software.
#
#     3.) Publication, Distribution, Sublicensing, and/or Selling of
#        copies or parts of the Software requires special agreements
#        with the University of Paderborn and is in general not permitted.
#
#     4.) Modifications or contributions to the software must be
#        published under this license. The University of Paderborn
#        is granted the non-exclusive right to publish modifications
#        or contributions in future versions of the Software free of charge.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#     OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#     NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#     WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#     OTHER DEALINGS IN THE SOFTWARE.
#
#     Persons using the Software are encouraged to notify the
#     Signal and System Theory Group at the University of Paderborn
#     about bugs. Please reference the Software in your publications
#     if it was used for them.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from mpl_toolkits.axes_grid1 import ImageGrid


def calculate_cov(X):
    """
    Calculate covariance matrices of given data.
    Parameters
    ----------
    X: np.ndarray
        dimensions: number components x number samples x number datasets

    Returns
    -------

    """

    cov = np.zeros((X.shape[2], X.shape[2], X.shape[0]))
    for comp_id in range(X.shape[0]):
        std = np.std(X[comp_id, :, :], axis=0, ddof=1)
        cov[:, :, comp_id] = np.cov(X[comp_id, :, :].T) / np.outer(std, std)
    return cov


def plot_scv_covs(scv_cov, n_cols=None):
    """
    Plot covariance matrix of each SCV.

    Parameters
    ----------
    scv_cov : np.ndarray
        Tensor with dimensions K x K x N, with K: number of datasets, and N: number of SCVs.
        scv_cov[:,:,i] represents covariance matrix of ith SCV

    filename : str, optional
        if filename is given, the figure is saved as filename.tex

    Returns
    -------
    None

    """

    n_sources = scv_cov.shape[2]
    if n_cols is None:
        n_cols = int(np.floor(np.sqrt(n_sources / 2) * 2))
    n_rows = int(np.ceil(n_sources / n_cols))

    fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(n_rows, n_cols),
                     axes_pad=0.4,
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.15,
                     )

    # Add data to image grid and plot
    for i, ax in enumerate(grid):
        if i < n_sources:
            im = ax.imshow(np.abs(scv_cov[:, :, i]), vmin=0, vmax=1, cmap='hot')
            ax.set_title(f'SCV {i + 1}', fontsize=20, pad=4)
        ax.axis('off')

    # Big colorbar on the right
    ax.cax.cla()
    cb.Colorbar(ax.cax, im)

    plt.show()
