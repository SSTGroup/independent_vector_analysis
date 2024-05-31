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
import scipy as sc
import time

from .helpers_iva import _normalize_column_vectors, whiten_data, _bss_isi, _decouple_trick, \
    _comp_l_sos_cost, _resort_scvs
from .initializations import _jbss_sos, _cca
from .iva_g import iva_g


def iva_l_sos(X, whiten=True, grad_projection=False, verbose=False, A=None, W_init=None,
              jdiag_initW=False, iva_g_initW = True, max_iter=2 * 512,
              termination_criterion='change_in_W',
              W_diff_stop=1e-4, alpha0=1.0, return_W_change=False):
    """
    Implementation of the independent vector analysis (IVA) algorithm using a correlated
    multivariate Laplace prior. This implementation also makes use of decoupling method to
    achieve 'Newton' like convergence (see references below).
    This implementation only works for real-valued data.
    For a general description of the algorithm and its relationship with others,
    see http://mlsp.umbc.edu/jointBSS_introduction.html.

    Parameters
    ----------
    X : np.ndarray
         data matrix of dimensions N x T x K.
         Real-valued data observations are from K data sets, i.e., X[k] = A[k] @ S[k], where A[k]
         is an N x N unknown invertible mixing matrix, and S[k] is N x T matrix with the nth row
         corresponding to T samples of the nth source in the kth dataset. This enforces the
         assumption of an equal number of samples in each dataset.
         For IVA, it is assumed that a source is statistically independent of all the other sources
         within the dataset and exactly dependent on at most one source in each of the other
         datasets.

    whiten : bool, optional
        If True, data is whitened. Whitening significantly decreases the runtime of the algorithm.
        If data is not zero-mean, whiten is forced to True

    grad_projection : bool, optional
        if True, the non-colinear direction is normalized

    verbose : bool, optional
         enables print statements if True

    A : np.ndarray, optional
         true mixing matrices A of dimensions N x N x K, automatically sets verbose to True

    W_init : np.ndarray, optional
         initial estimate for demixing matrices in W, with dimensions N x N x K

    jdiag_initW : bool, optional
        if True, use CCA (K=2) / joint diagonalization (K>2) for initialization, else random

    iva_g_initW : bool, optional
        If True, run IVA-G to find initialization for W, then run IVA-L-SOS

    max_iter : int, optional
         max number of iterations

    termination_criterion: str, optional
       if termination depends on W ('change_in_W') or on cost ('change_in_cost')

    W_diff_stop : float, optional
        termination threshold

    alpha0 : float, optional
         initial step size scaling (will be doubled for complex-valued gradient)

    return_W_change : bool, optional
        if the change in W in each iteration should be returned


    Returns
    -------
     W : np.ndarray
         the estimated demixing matrices of dimensions N x N x K so that ideally
         W[k] @ A[k] = P @ D[k], where P is any arbitrary permutation matrix and D[k] is any
         diagonal invertible (scaling) matrix. Note that P is common to all datasets. This is
         to indicate that the local permutation ambiguity between dependent sources
         across datasets should ideally be resolved by IVA.

     cost : float
         the cost for each iteration

     Sigma_N : np.ndarray
        Covariance matrix of each source component vector, with dimensions K x K x N

     isi : float
         joint inter-symbol-interference for each iteration. Only available if user supplies true
         mixing matrices for computing a performance metric. Else returns np.nan.


    Notes
    -----

    Coded by Matthew Anderson (matt.anderson@umbc.edu)
    Modified by Suchita Bhinge (suchita1@umbc.edu) on September 14 2018:
    Modification is in the score function in order to account for effect on
    large number of datasets
    Converted to Python by Isabell Lehmann (isabell.lehmann@sst.upb.de)

    References:

    [1] S. Bhinge, R. Mowakeaa, V. D. Calhoun, & T. Adali, "Extraction of time-varying
    spatio-temporal networks using parameter-tuned constrained IVA," Transaction on Medical Imaging,
    vol. 38, pp: 1715-1725, 2019
    [2] M. Anderson, T. Adali, & X.-L. Li,  "Joint Blind Source Separation of Multivariate Gaussian
    Sources: Algorithms and Performance Analysis," IEEE Trans. Signal Process., 2012, 60, 1672-1683
    [3] M. Anderson, G.-S. Fu, R. Phlypo, and T. Adali, "Independent vector analysis: Identification
    conditions and performance bounds," IEEE Trans. Signal Processing, vol. 62, pp. 4399-4410,
    Sep. 2014.

    Version 01 - 20120919 - Initial publication.
    Version 02 - 20140806 - Improved notation in comments.
    Version 03 - 20210301 - Python version.

    """

    start = time.time()

    if X.ndim != 3:
        raise AssertionError('X must have dimensions N x T x K.')
    elif X.shape[2] == 1:
        raise AssertionError('There must be ast least K=2 datasets.')

    if termination_criterion != 'change_in_W' and termination_criterion != 'change_in_cost':
        raise AssertionError("termination_criterion must be 'change_in_W' or 'change_in_cost'")

    # get dimensions
    N, T, K = X.shape

    if A is not None:
        supply_A = True
        if A.shape[0] != N or A.shape[1] != N or A.shape[2] != K:
            raise AssertionError('A must have dimensions N x N x K.')
    else:
        supply_A = False

    blowup = 1e3
    # set alpha0 to max(alpha_min, alpha0*alpha_scale) when cost does not decrease
    alpha_scale = 0.9
    alpha_min = W_diff_stop

    # test if data is zero-mean (test added by Isabell Lehmann)
    if np.linalg.norm(np.mean(X, axis=1)) > 1e-12:
        whiten = True

    if whiten:
        X, V = whiten_data(X)

    # calculate cross-covariance matrices of X (of shape N x N x K x K)
    R_xx = 1 / T * np.einsum('NTK, nTk -> NnKk', X, X)

    # Check rank of data-covariance matrix: should be full rank, if not we inflate (this is ad hoc)
    # concatenate all covariance matrices in a big matrix
    R_xx_all = np.moveaxis(R_xx, [0, 1, 2, 3], [0, 2, 1, 3]).reshape(
        (R_xx.shape[0] * R_xx.shape[2], R_xx.shape[1] * R_xx.shape[3]), order='F')
    rank = np.linalg.matrix_rank(R_xx_all)
    if rank < (N * K):
        # inflate Rx
        _, k, _ = np.linalg.svd(R_xx_all)
        R_xx_all += k[rank - 1] * np.eye(N * K)  # add smallest singular value to main diagonal
        R_xx = np.moveaxis(
            R_xx_all.reshape(R_xx.shape[0], R_xx.shape[2], R_xx.shape[1], R_xx.shape[3], order='F'),
            [0, 2, 1, 3], [0, 1, 2, 3])

    # Initializations

    # Initialize W
    if W_init is not None:
        if W_init.shape[0] != N or W_init.shape[1] != N or (
                W_init.ndim == 3 and W_init.shape[2] != K):
            raise AssertionError('W_init must have dimension N x N or N x N x K.')

        W = np.copy(W_init)

        if W.shape[0] == N and W.shape[1] == N and W.ndim == 2:
            W = np.tile(W[:, :, np.newaxis], [1, 1, K])

        if whiten:
            for k in range(K):
                W[:, :, k] = np.linalg.solve(V[:, :, k].T, W[:, :, k].T).T

    else:
        if jdiag_initW:
            if K > 2:
                # initialize with multi-set diagonalization (orthogonal solution)
                W = _jbss_sos(X, 0, 'whole')

            else:
                W = _cca(X)

        elif iva_g_initW:
            W = iva_g(X, max_iter=512)[0]

        else:
            # randomly initialize
            W = np.random.randn(N, N, K)
            if X.dtype == complex:
                W = W + 1j * np.random.randn(N, N, K)
            for k in range(K):
                W[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(W[:, :, k] @ W[:, :, k].T), W[:, :, k])

    if supply_A:
        isi = np.zeros(max_iter)
        if whiten:
            # A matrix is conditioned by V if data is whitened
            A_w = np.copy(A)
            for k in range(K):
                A_w[:, :, k] = V[:, :, k] @ A_w[:, :, k]
        else:
            A_w = np.copy(A)
    else:
        isi = None

    # Initialize some local variables
    cost = np.zeros(max_iter)
    gammaShape = (K + 1) ** 0.5  # .. Modified by Suchita Bhinge... September 14 2018
    Y = np.zeros_like(X)

    # to store the change in W or cost in each iteration
    W_change = []

    # Main Iteration Loop
    for iteration in range(max_iter):
        term_criterion = 0

        # Current estimated sources
        Y = np.einsum('NnK, nTK -> NTK', W, X)

        # Some additional computations of performance via ISI when True A is supplied
        if supply_A:
            avg_isi, joint_isi = _bss_isi(W, A_w)
            isi[iteration] = joint_isi

        W_old = np.copy(W)  # save current W as W_old
        cost[iteration], _ = _comp_l_sos_cost(W, Y)

        Q = 0
        R = 0

        # Loop over each SCV
        for n in range(N):

            # Efficient version of Sigma_n = 1/T * Y_n @ Y_n.T
            Sigma_n = np.eye(K, dtype=X.dtype)
            for k1 in range(K):
                for k2 in range(k1, K):
                    Sigma_n[k1, k2] = W[n, :, k1] @ R_xx[:, :, k1, k2] @ W[n, :, k2]
                    Sigma_n[k2, k1] = Sigma_n[k1, k2]

            hnk, Q, R = _decouple_trick(W, n, Q, R)

            # Derivative of cost function with respect to wnk
            yn = Y[n, :, :].T

            for k in range(K):
                Sigma_inv = np.linalg.inv(Sigma_n)
                gipyn = np.sum(yn * (Sigma_inv @ yn), axis=0)
                phi = (Sigma_inv[k, :] @ yn) * gipyn ** (
                    -0.5) * gammaShape  # .. Modified by Suchita Bhinge... September 14 2018
                dW = 1 / T * X[:, :, k] @ phi - hnk[:, k] / (W[n, :, k] @ hnk[:, k])
                if grad_projection:  # non-colinear direction normalized
                    dW = _normalize_column_vectors(dW - (W[n, :, k] @ dW) * W[n, :, k])
                W[n, :, k] = _normalize_column_vectors(W[n, :, k] - alpha0 * dW)

                yn[k, :] = W[n, :, k] @ X[:, :, k]
                for kk in range(K):
                    Sigma_n[k, kk] = W[n, :, k] @ R_xx[:, :, k, kk] @ W[n, :, kk]
                Sigma_n[:, k] = Sigma_n[k, :].T

        # Calculate termination criterion
        if termination_criterion == 'change_in_W':
            for k in range(K):
                term_criterion = np.maximum(term_criterion, np.amax(
                    1 - np.abs(np.diag(W_old[:, :, k] @ W[:, :, k].T))))

        elif termination_criterion == 'change_in_cost':
            if iteration == 0:
                term_criterion = 1
            else:
                term_criterion = np.abs(cost[iteration - 1] - cost[iteration]) / np.abs(
                    cost[iteration])

        W_change.append(term_criterion)

        # Decrease step size alpha if cost increased from last iteration
        if iteration > 0 and cost[iteration] > cost[iteration - 1]:
            alpha0 = np.maximum(alpha_min, alpha_scale * alpha0)

        # Check the termination condition
        if term_criterion < W_diff_stop or iteration == max_iter - 1:
            break
        elif term_criterion > blowup or np.isnan(cost[iteration]):
            for k in range(K):
                W[:, :, k] = np.eye(N) + 0.1 * np.random.randn(N, N)
            if X.dtype == complex:
                W = W + 0.1j * np.random.randn(N, N, K)
            if verbose:
                print('W blowup, restart with new initial value.')

        # Display Iteration Information
        if verbose:
            if supply_A:
                print(f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}, '
                      f'Avg ISI: {avg_isi}, Joint ISI: {joint_isi}')
            else:
                print(f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}')

    # Finish Display
    if iteration == 0 and verbose:
        if supply_A:
            print(
                f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}, '
                f'Avg ISI: {avg_isi}, Joint ISI: {joint_isi}')
        else:
            print(f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}')

    # Clean-up Outputs
    cost = cost[0:iteration + 1]

    if supply_A:
        isi = isi[0:iteration + 1]

    if whiten:
        for k in range(K):
            W[:, :, k] = W[:, :, k] @ V[:, :, k]
    else:  # no prewhitening
        # Scale demixing vectors to generate unit variance sources
        for n in range(N):
            for k in range(K):
                W[n, :, k] /= np.sqrt(W[n, :, k] @ R_xx[:, :, k, k] @ W[n, :, k])

    # Resort order of SCVs: Order the components from most to least ill-conditioned
    if not whiten:
        V = None
    W, Sigma_N = _resort_scvs(W, R_xx, whiten, V)

    end = time.time()

    if verbose:
        print(f"IVA-L-SOS finished after {(end - start) / 60} minutes with {iteration} iterations.")

    if return_W_change:
        return W, cost, Sigma_N, isi, W_change
    else:
        return W, cost, Sigma_N, isi
