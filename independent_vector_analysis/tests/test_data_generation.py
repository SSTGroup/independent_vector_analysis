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

from ..data_generation import generate_sources, MGGD_generation, randmv_laplace


def test_generate_sources():
    N = 4
    K = 3
    T = 1000

    # generating different rho value for each source
    rho_mean = 0.8
    rho_range = 0.15
    rho = np.linspace(rho_mean - rho_range, rho_mean + rho_range, N)

    S_matrix = generate_sources(rho, N, T, K)

    # calculate covariance matrix for each source component vector
    for n in range(N):
        R = 1 / T * S_matrix[n, :, :].T @ S_matrix[n, :, :]
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if i != j:
                    # should be approximalety the same as the true correlation
                    np.testing.assert_almost_equal(rho[n], R[i, j], decimal=1)

    # check that no error is raised when called with an int value
    generate_sources(0.8, N, T, K)

    # check that error is raised when called with rho, which does not has length N
    try:
        generate_sources(rho[:-2], N, T, K)
    except:
        print('Error is raised for wrong shape of rho')


def test_mggd():
    X = MGGD_generation(1000, 7, 'uniform', 0.5, 1)
    X = MGGD_generation(1000, 7, 'ar', 0.5, 1)
    rho = {'val': np.array([0.5, 0.8]), 'idx': np.array([0, 2, 3, 4])}
    X = MGGD_generation(1000, 7, 'two_rho', rho, 1)
    X = MGGD_generation(1000, 7, 'q_qt', 0.5, 1)
    X = MGGD_generation(1000, 7, 'rho_list', np.array([0.8, 0.7, 0.7, 0.8, 0.7, 0.8]), 1)
    X = MGGD_generation(1000, 7, 'rho_list', [0.8, 0.7, 0.7, 0.8, 0.7, 0.8], 1)
    X = MGGD_generation(1000, 7, 'block', {'val': 0.3, 'blocks': (0.9, 4, 8)}, 1)


def test_randmv_laplace():
    T = int(1e6)
    dim = 2
    Nbins = [100, 100]

    gamma = np.array([[1, 0.5], [0.5, 1]])
    gamma = gamma / np.sqrt(np.linalg.det(gamma))
    lambda_ = 1
    mu = np.array([0, 2])
    Z = randmv_laplace(dim, T, gamma=gamma, lambda_=lambda_, mu=mu)

    plt.figure()
    hist_out, xedges, yedges, img = plt.hist2d(Z[0], Z[1], bins=Nbins)
    plt.imshow(hist_out)
    plt.colorbar()
    plt.xlabel('Z[0]')
    plt.ylabel('Z[1]')
    plt.title('Histogram of Multivariate Laplace')
    plt.show()
