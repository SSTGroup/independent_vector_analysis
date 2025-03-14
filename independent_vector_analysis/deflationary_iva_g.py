import numpy as np
import scipy as sc
import time

from .helpers_iva import _normalize_column_vectors, _decouple_trick, _bss_isi, whiten_data, _resort_scvs
from .initializations import _jbss_sos, _cca


def deflationary_iva_g(X, whiten=True,
                       verbose=False, A=None, W_init=None, jdiag_initW=False, max_iter=1024,
                       W_diff_stop=1e-6, alpha0=1.0, return_W_change=False, R_xx=None, update='newton'):
    """
    Implementation of all the second-order (Gaussian) independent vector analysis (IVA) algorithms.
    Namely real-valued and complex-valued with circular and non-circular using Newton, gradient,
    and quasi-Newton optimizations.
    For a general description of the algorithm and its relationship with others,
    see http://mlsp.umbc.edu/jointBSS_introduction.html.


    Parameters
    ----------
    X : np.ndarray
        data matrix of dimensions N x T x K.
        Data observations are from K data sets, i.e., X[k] = A[k] @ S[k], where A[k] is an N x N
        unknown invertible mixing matrix, and S[k] is N x T matrix with the nth row corresponding to
        T samples of the nth source in the kth dataset. This enforces the assumption of an equal
        number of samples in each dataset.
        For IVA, it is assumed that a source is statistically independent of all the other sources
        within the dataset and exactly dependent on at most one source in each of the other
        datasets.

    whiten : bool, optional
        if True, data is whitened.
        For the complex-valued gradient and for quasi approach, whiten is forced to True.
        If data is not zero-mean, whiten is forced to True

    verbose : bool, optional
        enables print statements if True

    A : np.ndarray, optional
        true mixing matrices A of dimensions N x N x K, automatically sets verbose to True

    W_init : np.ndarray, optional
        initial estimate for demixing matrices in W, with dimensions N x N x K

    jdiag_initW : bool, optional
        if True, use CCA (K=2) / joint diagonalization (K>2) for initialization, else random

    max_iter : int, optional
        max number of iterations

    W_diff_stop : float, optional
        stopping criterion

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

    j_isi : float
        joint isi (inter-symbol-interference) for each iteration. Only available if user
        supplies true mixing matrices for computing a performance metric. Else returns np.nan.


    Notes
    -----

    Coded by Matthew Anderson (matt.anderson@umbc.edu)
    Converted to Python by Isabell Lehmann (isabell.lehmann@sst.upb.de)

    References:

    [1] M. Anderson, X.-L. Li, & T. Adalı, "Nonorthogonal Independent Vector Analysis Using
    Multivariate Gaussian Model," LNCS: Independent Component Analysis and Blind Signal Separation,
    Latent Variable Analysis and Signal Separation, Springer Berlin / Heidelberg, 2010, 6365,
    354-361
    [2] M. Anderson, T. Adalı, & X.-L. Li, "Joint Blind Source Separation of Multivariate Gaussian
    Sources: Algorithms and Performance Analysis," IEEE Trans. Signal Process., 2012, 60, 1672-1683
    [3] M. Anderson, X.-L. Li, & T. Adalı, "Complex-valued Independent Vector Analysis: Application
    to Multivariate Gaussian Model," Signal Process., 2012, 1821-1831

    Version 01 - 20120913 - Initial publication
    Version 02 - 20120919 - Using decouple_trick function, bss_isi, whiten_data, & cca subfunctions
    Version 03 - 20210129 - Python version of the code, with max_iter=1024.

    """

    start = time.time()

    if X.ndim != 3:
        raise AssertionError('X must have dimensions N x T x K.')
    elif X.shape[2] == 1:
        raise AssertionError('There must be ast least K=2 datasets.')

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

    if R_xx is None:
        # calculate cross-covariance matrices of X
        R_xx = np.zeros((N, N, K, K), dtype=X.dtype)
        for k1 in range(K):
            for k2 in range(k1, K):
                R_xx[:, :, k1, k2] = 1 / T * X[:, :, k1] @ X[:, :, k2].T
                R_xx[:, :, k2, k1] = R_xx[:, :, k1, k2].T  # R_xx is symmetric

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

        else:
            # randomly initialize
            W = np.random.randn(N, N, K)
            for k in range(K):
                W[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(W[:, :, k] @ W[:, :, k].T), W[:, :, k])

    if supply_A:
        j_isi = np.zeros(1)
        if whiten:
            # A matrix is conditioned by V if data is whitened
            A_w = np.copy(A)
            for k in range(K):
                A_w[:, :, k] = V[:, :, k] @ A_w[:, :, k]
        else:
            A_w = np.copy(A)
    else:
        j_isi = None

    # Initialize some local variables
    cost = np.zeros(max_iter)
    cost_const = K * np.log(2 * np.pi * np.exp(1))  # local constant

    # to store the change in W in each iteration
    W_change = []

    # Loop over each SCV
    for n in range(N):

        if verbose:
            print(f'SCV {n + 1}')

        # Main Iteration Loop
        for iteration in range(max_iter):

            W_old = np.copy(W)  # save current W as W_old
            cost[iteration] = 0
            for k in range(K):
                cost[iteration] -= np.log(np.abs(np.linalg.det(W[:, :, k])))

            # Efficient version of Sigma_n = 1/T * Y_n @ Y_n.T with Y_n = W_n @ X_n
            Sigma_n = np.eye(K)

            for k1 in range(K):
                for k2 in range(k1, K):
                    Sigma_n[k1, k2] = W[n, :, k1] @ R_xx[:, :, k1, k2] @ W[n, :, k2]
                    Sigma_n[k2, k1] = Sigma_n[k1, k2]

            cost[iteration] += 0.5 * (cost_const + np.log(np.abs(np.linalg.det(Sigma_n))))

            Sigma_inv = np.linalg.inv(Sigma_n)

            hnk = _decouple_trick(W, n)

            # compute gradient
            grad = np.zeros((N, K))
            norm_grad = np.zeros((N, K))
            for k in range(K):
                # Analytic derivative of cost function with respect to vn
                # Code below is efficient implementation of computing the gradient, which is
                # independent of T
                grad[:, k] = - hnk[:, k] / (W[n, :, k] @ hnk[:, k])

                for kk in range(K):
                    grad[:, k] += R_xx[:, :, k, kk] @ W[n, :, kk] * Sigma_inv[kk, k]

                temp_grad = (np.eye(N) - np.outer(W[n, :, k], W[n, :, k])) @ grad[:, k]
                norm_grad[:, k] = temp_grad / np.linalg.norm(temp_grad)

            # Compute SCV Hessian
            H = np.zeros((N * K, N * K))
            for k1 in range(K):
                H[k1 * N:k1 * N + N, k1 * N:k1 * N + N] = \
                    Sigma_inv[k1, k1] * R_xx[:, :, k1, k1] + np.outer(
                        hnk[:, k1], hnk[:, k1]) / (hnk[:, k1] @ W[n, :, k1]) ** 2

                for k2 in range(k1 + 1, K):
                    Hs = Sigma_inv[k1, k2] * R_xx[:, :, k2, k1].T
                    H[k1 * N: k1 * N + N, k2 * N: k2 * N + N] = Hs
                    H[k2 * N: k2 * N + N, k1 * N: k1 * N + N] = Hs.T

            # update w_n^[1] ... w_n^[K]
            Wn = W[n, :, :].flatten(order='F')
            if update == 'newton':
                Wn -= alpha0 * np.linalg.solve(H, grad.flatten('F'))
            elif update == 'gradient':
                Wn -= alpha0 * grad.flatten('F')
            elif update == 'norm_gradient':
                Wn -= alpha0 * norm_grad.flatten('F')

            # Store Updated W
            Wn = np.reshape(Wn, (N, K), 'F')
            for k in range(K):
                W[n, :, k] = Wn[:, k]

            # make demixing vector w_n^[k] orthogonal to all previous w_1^[k] ... w_{n-1}^[k]
            for k in range(K):
                if n > 0:
                    Wnk = W[0:n, :, k]  # (n-1) x N matrix containing n-1 previous demixing vectors
                    Pnk = np.eye(N) - Wnk.T @ np.linalg.inv(Wnk @ Wnk.T) @ Wnk
                    W[n, :, k] = Pnk @ W[n, :, k]  # update w_i^[k]
                W[n, :, k] /= np.linalg.norm(W[n, :, k])  # make vectors unit-norm

            # make demixing vectors w_{n+1}^[k], ..., w_N^[k] orthogonal to w_1^[k], ..., w_n^[k]
            if n < N - 1:
                for k in range(K):
                    Wn1k = W[0:n + 1, :, k]  # n x N matrix containing n previous demixing vectors
                    Pn1k = np.eye(N) - Wn1k.T @ np.linalg.inv(Wn1k @ Wn1k.T) @ Wn1k
                    W[n + 1:, :, k] = W[n + 1:, :, k] @ Pn1k  # update w_{n+1}^[k], ..., w_N^[k]
                    # make w_{n+1}^[k], ..., w_N^[k] orthogonal to each other
                    W[n + 1:, :, k] = np.linalg.solve(sc.linalg.sqrtm(W[n + 1:, :, k] @ W[n + 1:, :, k].T),
                                                      W[n + 1:, :, k])

            term_criterion = 0
            for k in range(K):
                term_criterion = np.maximum(term_criterion, 1 - np.abs(W_old[n, :, k] @ W[n, :, k].T))

            W_change.append(term_criterion)

            # Decrease step size alpha if cost increased from last iteration
            if iteration > 0 and cost[iteration] > cost[iteration - 1]:
                alpha0 = np.maximum(alpha_min, alpha_scale * alpha0)

            # Check the termination condition
            if term_criterion < W_diff_stop:
                break
            elif term_criterion > blowup or np.isnan(cost[iteration]):
                for k in range(K):
                    W[:, :, k] = np.eye(N) + 0.1 * np.random.randn(N, N)
                if verbose:
                    print('W blowup, restart with new initial value.')

            # Display Iteration Information
            if verbose:
                print(f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}')

        # Finish Display
        if iteration == 0 and verbose:
            print(f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}')

    # Clean-up Outputs
    cost = cost[0:iteration + 1]

    # Some additional computations of performance via ISI when true A is supplied
    if supply_A:
        avg_isi, joint_isi = _bss_isi(W, A_w)
        j_isi = joint_isi[np.newaxis]

    if whiten:
        for k in range(K):
            W[:, :, k] = W[:, :, k] @ V[:, :, k]
    else:  # no prewhitening
        # Scale demixing vectors to generate unit variance sources
        for n in range(N):
            for k in range(K):
                W[n, :, k] /= np.sqrt(W[n, :, k] @ R_xx[:, :, k, k] @ W[n, :, k])

    # Resort order of SCVs: Order the components from most to least ill-conditioned
    P_xx = None
    if not whiten:
        V = None
    W, Sigma_N = _resort_scvs(W, R_xx, whiten, V, False, False, P_xx)

    end = time.time()

    if verbose:
        print(f"IVA-G finished after {(end - start) / 60} minutes with {iteration} iterations.")

    if return_W_change:
        return W, cost, Sigma_N, j_isi, W_change
    else:
        return W, cost, Sigma_N, j_isi
