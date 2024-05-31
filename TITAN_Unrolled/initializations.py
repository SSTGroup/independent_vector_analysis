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
import scipy.linalg


def _jbss_sos(x, tau=0, cost_selection='cross'):
    """
    Joint blind source separation using second order statistics.


    Parameters
    ----------
    x : np.ndarray
        the mixtures of dimensions N x T x K

    tau : int or np.ndarray
        the non-negative time delays, e.g. 0 or np.arange(5) or np.array([0,3,7,12]).
        Always assume tau[0]=0

    cost_selection : str, optional
        'cross' or 'whole'. In case of 'cross', only the cross cost is minimized by using
        sequential orthogonal procrustes. In case of 'whole', a gradient search algorithm is used,
        starting from the initial guess provided by sequential orthogonal procrustes.


    Returns
    -------
    W : np.ndarray
        demixing matrix of dimensions N x N x K


    Notes
    -----
    For a general description of the algorithm and its relationship with others,
    see http://mlsp.umbc.edu/jointBSS_introduction.html

    References:
    [1] X.-L. Li, T. Adali, & M. Anderson, "Joint Blind Source Separation by Generalized Joint
    Diagonalization of Cumulant Matrices," Signal Process., 2011, 91, 2314-2322
    [2] X.-L. Li, M. Anderson, & T. Adali, "Second and Higher-Order Correlation Analysis of
    Multiset Multidimensional Variables by Joint Diagonalization," Lecture Notes in Computer
    Science: Independent Component Analysis and Blind Signal Separation, Latent Variable Analysis
    and Signal Separation, Springer Berlin / Heidelberg, 2010, 6365, 197-204

    Code is converted to Python by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    """

    N, T, K = x.shape

    if type(tau) is not int:
        tau_ = np.copy(tau)
        tau_[0] = 0
    else:
        tau_ = np.expand_dims(tau, axis=0)
    tau_ = np.abs(tau_)
    L = tau_.size

    # first remove mean
    x_w = np.zeros_like(x)
    for k in range(K):
        for n in range(N):
            x_w[n, :, k] = x[n, :, k] - np.mean(x[n, :, k])

    # second whitening the mixtures
    P = np.zeros((N, N, K), dtype=x.dtype)
    for k in range(K):
        R = 1 / T * x_w[:, :, k] @ np.conj(x_w[:, :, k].T)
        P[:, :, k] = np.linalg.inv(sc.linalg.sqrtm(R))
        x_w[:, :, k] = P[:, :, k] @ x_w[:, :, k]

    # third construct the correlation matrix at L time-delays
    C = np.zeros((N, N, K, K, L), dtype=x.dtype)
    for k1 in range(K):
        for k2 in range(K):
            for el in range(L):
                C[:, :, k1, k2, el] = 1 / T * x_w[:, 0:T - tau_[el], k1] @ np.conj(
                    x_w[:, tau_[el]:T, k2].T)

    # joint diagonalization
    W0 = np.zeros((N, N, K), dtype=x.dtype)
    for k in range(K):
        W0[:, :, k] = np.eye(N)

    if cost_selection == 'whole':
        W0 = _diag_cross(C, W0, 1e-8)
        W = _diag_whole(C, W0)

    elif cost_selection == 'cross':
        if L == 1:
            W = _sym_diag_cross(C, W0, 1e-8)
        else:
            W = _diag_cross(C, W0, 1e-8)

    else:
        raise ValueError("cost_selection must be 'whole' or 'cross'")

    # recovery the whitening
    for k in range(K):
        W[:, :, k] = W[:, :, k] @ P[:, :, k]

    return W


def _diag_cross(T, W0=None, tol=1e-10):
    """
    Joint diagonalization algorithm minimizing the cross cost by sequential orthogonal procrustes.


    Parameters
    ----------
    T : np.ndarray
        target matrices of dimensions N x N x K x K x L

    W0 : np.ndarray, optional
        initial guess of demixing matrix of dimensions N x N x K

    tol : float, optional
        stopping condition


    Returns
    -------
    W : np.ndarray
        demixing matrix of dimensions N x N x K

    Code is converted to Python by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    """

    N, _, K, _, L = T.shape
    D = np.zeros((N, K, K, L), dtype=T.dtype)

    if W0 is None:
        W0 = np.zeros((N, N, K))
        for k in range(K):
            W0[:, :, k] = np.random.randn(N, N)
            W0[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(W0[:, :, k] @ W0[:, :, k].T), W0[:, :, k])
    W = np.copy(W0)

    A = np.zeros((2 * L * N * (K - 1), N, K), dtype=T.dtype)
    for p in range(K):
        cnt = 0
        for q in range(K):
            if q != p:
                for el in range(L):
                    A[cnt * N:(cnt + 1) * N, :, p] = np.conj(T[:, :, p, q, el].T)
                    cnt = cnt + 1
                    A[cnt * N:(cnt + 1) * N, :, p] = T[:, :, q, p, el]
                    cnt = cnt + 1

    max_iter = 1000
    cost = np.zeros(max_iter)
    last_W = np.copy(W)

    for iter in range(max_iter):

        cost[iter] = 0
        for p in range(K):
            W_p = W[:, :, p]
            for q in range(K):
                if q != p:
                    W_qh = np.conj(W[:, :, q].T)
                    for el in range(L):
                        D[:, p, q, el] = np.diag(W_p @ T[:, :, p, q, el] @ W_qh)

        B = np.zeros((2 * L * N * (K - 1), N), dtype=T.dtype)
        randp = np.random.permutation(K)
        for p0 in range(K):
            p = randp[p0]

            cnt = 0
            for q in range(K):
                if q != p:
                    for el in range(L):
                        B[cnt * N:(cnt + 1) * N, :] = np.conj(W[:, :, q].T) @ np.diag(
                            np.conj(D[:, p, q, el]))
                        cnt += 1
                        B[cnt * N: (cnt + 1) * N, :] = np.conj(W[:, :, q].T) @ np.diag(
                            D[:, q, p, el])
                        cnt += 1
            U, _, Vh = np.linalg.svd(np.conj(B.T) @ A[:, :, p])
            W[:, :, p] = U @ Vh

        tol_ = 0
        for k in range(K):
            tol_ = np.maximum(tol_, np.amax(1 - np.diag(W[:, :, k] @ np.conj(last_W[:, :, k].T))))
        if tol_ < tol:
            break
        else:
            last_W = np.copy(W)

    return W


def _diag_whole(T, W0, tol=1e-5):
    """
    Joint diagonalization algorithm minimizing the whole cost


    Parameters
    ----------
    T : np.ndarray
        target matrices of dimensions N x N x K x K x L

    W0 : np.ndarray
        initial guess of demixing matrix of dimensions N x N x K

    tol : float, optional
        stopping condition


    Returns
    -------
    W : np.ndarray
        demixing matrix of dimensions N x N x K

    Code is converted to Python by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    """

    W = np.copy(W0)
    mu = 1 / 20
    W = _iva_diag_kernel(T, W, mu, tol)
    return W


def _iva_diag_kernel(T, W0, mu, tol=1e-5):
    """
    The actual gradient search algorithm

    Parameters
    ----------
    T : np.ndarray
        target matrices of dimensions N x N x K x K x L

    W0 : np.ndarray
        initial guess of demixing matrix of dimensions N x N x K

    mu : float
        step size

    tol : float, optional
        stopping condition


    Returns
    -------
    W : np.ndarray
        demixing matrix of dimensions N x N x K

    Code is converted to Python by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    """

    N, _, K, _, L = T.shape

    max_iter = 1000
    cost = np.zeros(max_iter)
    min_cost = np.inf
    best_W = np.copy(W0)
    last_W = np.copy(W0)
    W = np.copy(W0)
    cost_increase_counter = 0
    # tol = 1
    for iter in range(max_iter):

        cost[iter] = 0
        for p in range(K):
            for q in range(K):
                for el in range(L):
                    B = W[:, :, p] @ T[:, :, p, q, el] @ np.conj(W[:, :, q].T)
                    B = B - np.diag(np.diag(B))
                    cost[iter] += np.linalg.norm(B) ** 2

        grad = np.zeros((N, N, K), dtype=T.dtype)
        for p in range(K):
            for el in range(L):
                for q in range(K):
                    B = W[:, :, p] @ T[:, :, p, q, el] @ np.conj(W[:, :, q].T)
                    B = B - np.diag(np.diag(B))
                    grad[:, :, p] = grad[:, :, p] + B @ W[:, :, q] @ np.conj(T[:, :, p, q, el].T)

                    B = W[:, :, p] @ np.conj(T[:, :, q, p, el].T) @ np.conj(W[:, :, q].T)
                    B = B - np.diag(np.diag(B))
                    grad[:, :, p] = grad[:, :, p] + B @ W[:, :, q] @ T[:, :, q, p, el]

            grad[:, :, p] -= W[:, :, p] @ np.conj(grad[:, :, p].T) @ W[:, :, p]
            grad[:, :, p] /= np.linalg.norm(grad[:, :, p], ord=2)  # normalization

            W[:, :, p] = W[:, :, p] - mu * grad[:, :, p]
            if W.dtype == complex:
                W[:, :, p] = np.linalg.solve(sc.linalg.sqrtm(W[:, :, p] @ np.conj(W[:, :, p].T)),
                                             W[:, :, p])
            else:  # use real part of sqrtm since imaginary is 0
                W[:, :, p] = np.linalg.solve(np.real(sc.linalg.sqrtm(W[:, :, p] @ W[:, :, p].T)),
                                             W[:, :, p])

        if cost[iter] < min_cost:
            min_cost = cost[iter]
            best_W = np.copy(last_W)

        if cost[iter] > min_cost:
            cost_increase_counter += 1
            if cost_increase_counter > 1:
                cost_increase_counter = 0
                mu = 0.5 * mu
                W = np.copy(best_W)
                last_W = np.copy(W)
                continue  # skip code below and jump to next iteration

        tol_ = 0
        for k in range(K):
            tol_ = np.maximum(tol_, np.amax(1 - np.diag(W[:, :, k] @ np.conj(last_W[:, :, k].T))))
        if tol_ < tol:
            break
        else:
            last_W = np.copy(W)

    W = np.copy(best_W)

    return W


def _sym_diag_cross(T, W0=None, tol=1e-10):
    """
    Joint diagonalization algorithm minimizing the cross cost by sequential orthogonal procrustes.
    Assume that T is symmetric, i.e., T[:,:,k1,k2,:] = T[:,:,k2,k1,:].T, and only operate on
    T[:,:,k1,k2,:], k2 > k1


    Parameters
    ----------
    T : np.ndarray
        target matrices of dimensions N x N x K x K x L

    W0 : np.ndarray
        initial guess of demixing matrix of dimensions N x N x K

    tol : float, optional
        stopping condition


    Returns
    -------
    W : np.ndarray
        demixing matrix of dimensions N x N x K

    Code is converted to Python by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    """

    N, _, K, _, L = T.shape
    D = np.zeros((N, K, K, L), dtype=T.dtype)

    if W0 is None:
        W0 = np.zeros((N, N, K), dtype=T.dtype)
        for k in range(K):
            if T.dtype == complex:
                W0[:, :, k] = np.random.randn(N, N) + 1j * np.random.randn(N, N)
            else:
                W0[:, :, k] = np.random.randn(N, N)

            W0[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(W0[:, :, k] * W0[:, :, k].T), W0[:, :, k])

    W = np.copy(W0)

    max_iter = 1000
    cost = np.zeros(max_iter)
    last_W = np.copy(W)

    for iter in range(max_iter):
        cost[iter] = 0
        for p in range(K):
            W_p = W[:, :, p]
            for q in range(p + 1, K):
                W_qh = np.conj(W[:, :, q].T)
                for el in range(L):
                    D[:, p, q, el] = np.diag(W_p @ T[:, :, p, q, el] @ W_qh)

        A = np.zeros((L * N * (K - 1), N), dtype=T.dtype)
        B = np.zeros((L * N * (K - 1), N), dtype=T.dtype)
        randp = np.random.permutation(K)
        for p0 in range(K):
            p = randp[p0]

            cnt = 0
            for q in range(p + 1, K):
                for el in range(L):
                    A[cnt * N:(cnt + 1) * N, :] = np.conj(T[:, :, p, q, el].T)
                    B[cnt * N:(cnt + 1) * N, :] = np.conj(W[:, :, q].T) @ np.diag(
                        np.conj(D[:, p, q, el].T))
                    cnt = cnt + 1
            for q in range(p):
                for el in range(L):
                    A[cnt * N:(cnt + 1) * N, :] = T[:, :, q, p, el]
                    B[cnt * N:(cnt + 1) * N, :] = np.conj(W[:, :, q].T) @ np.diag(D[:, q, p, el])
                    cnt = cnt + 1

            U, _, Vh = np.linalg.svd(np.conj(B.T) @ A)
            W[:, :, p] = U @ Vh

        tol_ = 0
        for k in range(K):
            tol_ = np.maximum(tol_, np.amax(1 - np.diag(W[:, :, k] @ np.conj(last_W[:, :, k].T))))
        if tol_ < tol:
            break
        else:
            last_W = np.copy(W)

    return W


def _cca(X):
    """
    Performs canonical correlation analysis.


    Parameters
    ----------
    X : np.ndarray
        data of two datasets, with dimensions N x T x 2


    Returns
    -------
    W : np.ndarray
        transformation matrices for the two datasets, with dimensions N x N x 2

    Code is converted to Python by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    """

    N, T, K = X.shape

    if K != 2:
        raise ValueError('CCA only works for two datasets')

    R_xx = np.zeros((N, N, 2, 2), dtype=X.dtype)
    for k1 in range(2):
        for k2 in range(2):
            R_xx[:, :, k1, k2] = 1 / T * X[:, :, k1] @ X[:, :, k2].T

    temp = np.linalg.solve(R_xx[:, :, 1, 1], R_xx[:, :, 1, 0])
    A = R_xx[:, :, 0, 1] @ temp
    _, W0 = sc.linalg.eig(A, R_xx[:, :, 0, 0])
    W1 = temp @ W0
    W = np.zeros((N, N, 2), dtype=X.dtype)
    W[:, :, 0] = W0.T
    W[:, :, 1] = W1.T

    return W
