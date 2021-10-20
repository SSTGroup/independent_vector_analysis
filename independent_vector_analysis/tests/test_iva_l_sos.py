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
from ..iva_l_sos import iva_l_sos


def test_iva_real_function_calls():
    """
    Make sure that function calls do not raise errors. Final value is not that important, therefore
    max-iter is set to 4.
    """

    N = 20  # number of sources
    T = 1000  # sample size
    K = 40  # number of groups

    # generate the mixtures
    S = np.zeros((N, T, K))
    for n in range(N):
        temp1 = np.random.randn(K, T)
        temp = np.zeros((K, T))
        B = np.random.randn(K, K, 3)

        for p in range(2):
            for t in range(2, T):
                # introduce nonwhiteness and spatial correlation
                temp[:, t] += B[:, :, p] @ temp1[:, t - p]

        for k in range(K):
            S[n, :, k] = temp[k, :]
            S[n, :, k] -= np.mean(S[n, :, k])
            S[n, :, k] = S[n, :, k] / np.std(S[n, :, k], ddof=1)

    A = np.random.randn(N, N, K)
    X = np.zeros((N, T, K))
    for k in range(K):
        X[:, :, k] = A[:, :, k] @ S[:, :, k]

    # initialize with multi-set diagonalization
    W, _, _, isi = iva_l_sos(X, verbose=True, A=A, jdiag_initW=True, max_iter=4)

    # CCA init (for 2 datasets)
    W, _, _, isi = iva_l_sos(X[:, :, 0:2], verbose=True, A=A[:, :, 0:2], jdiag_initW=True,
                             max_iter=4)

    # W_init is given
    W_init = np.zeros_like(A)
    for k in range(K):
        W_init[:, :, k] = np.linalg.inv(A[:, :, k]) + np.random.randn(N, N) * 0.1

    W, _, _, isi = iva_l_sos(X, verbose=True, A=A, W_init=W_init, max_iter=4)

    # same W_init for each dataset
    W, _, _, isi = iva_l_sos(X, verbose=True, A=A, W_init=W_init[:, :, 0], max_iter=4)

    # random init
    W, _, _, isi = iva_l_sos(X, verbose=True, A=A, max_iter=4)

    # no whitening
    W, _, _, isi = iva_l_sos(X, whiten=False, verbose=True, A=A, max_iter=4)

    # with grad projection
    W, _, _, isi = iva_l_sos(X, grad_projection=True, verbose=True, A=A, max_iter=4)

    # no whitening with grad projeciton
    W, _, _, isi = iva_l_sos(X, whiten=False, grad_projection=True, verbose=True, A=A, max_iter=4)

    # change step size
    W, _, _, isi = iva_l_sos(X, verbose=True, A=A, max_iter=4, alpha0=2)

    # test stopping criterion
    W, _, _, isi = iva_l_sos(X, verbose=True, A=A, W_init=W_init)
    print(f'IVA-L-SOS ISI for random init for stop on W: {isi}')

    # change stopping criterion
    W, _, _, isi = iva_l_sos(X, verbose=True, A=A, W_init=W_init,
                             termination_criterion='change_in_cost')
    print(f'IVA-L-SOS ISI for random init for stop on cost: {isi}')

