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
import scipy.linalg
from scipy.special import gamma


def normalize_row_vectors(x):
    return x / np.linalg.norm(x, axis=1)[:, np.newaxis]


def generate_sources(rho, N, T, K):
    """
    Generate sources with simple correlation structure:
    For i = 1, ..., N, the ith source in each of the K datasets is correlated.

    Parameters
    ----------

    rho : float or np.ndarray
        correlation value or array of dimension N containing the correlation value for each SCV

    N : int
        number of sources in one source vector

    T : int
        number of samples in each source

    K : int
        number of datasets

    Returns
    -------
    S : np.ndarray
        matrix of dimensions N x T x K, with S[n,:,k] is the nth source in the kth dataset

    """

    # fit shape of rho
    if type(rho) == float:
        rho = np.ones(N) * rho
    else:
        assert (N == rho.shape[0]), 'rho has incorrect shape, must have shape N'

    # joint mean
    mu = np.zeros(N * K)

    # joint covariance matrix
    R = np.eye(K * N)
    for i in range(K):
        for j in range(i, K):
            # cross covariance matrices
            if i != j:
                R[i * N:(i + 1) * N, j * N:(j + 1) * N] = np.diag(rho)
                R[j * N:(j + 1) * N, i * N:(i + 1) * N] = R[i * N:(i + 1) * N, j * N:(j + 1) * N]

    # draw source vector of joint distribution
    all_sources = np.random.multivariate_normal(mu, R, T).T

    S_matrix = np.zeros((N, T, K))
    for i in range(N * K):
        n = np.mod(i, N)
        k = i // N
        S_matrix[n, :, k] = all_sources[i, :]

    return S_matrix


def MGGD_generation(N, dim=None, correlation_structure=None, rho=None, beta=1, cov=None):
    """
    Contains six functions that can be used to generate multivariate generalized Gaussian
    distributed (MGGD) sources. If cov is given, none of the functions will be used but data will
    be generated using cov.
    
    
    Parameters
    ----------
    N : int
        Number of realizations / samples
        
    dim : int, optional
        Dimension / number of variables. Must be given if cov is None.

    correlation_structure : str, optional
        Which structure of correlation matrix to use. Must be given if cov is None.
            - 'uniform': elements are equal to rho
            - 'ar' (auto regressive): cov[i, j] = rho ** (np.abs(i - j))
            - 'q_qt': normalized random samples ~ U[-0.2, 0.8) multiplied by themselves
            - 'two_rho': most elements are equal, specific elements have another value
            - 'rho_list': elements of each row/column of the covariance matrix have a specific value
            - 'block': block-diagonal matrix. Elements in a block have same value, non
                       block-elements have same value, diagonal elements are 1
        
    rho : float or dict or np.ndarray or list, optional
        Correlation value. Must be given if cov is None and correlation_structure is not 'q_qt'.
        If correlation_structure=='two_rho', rho must be a dictionary with keys 'val' and 'idx'.
            'val' is a np.ndarray or list with two values of correlation: 'val[0]' provides the
            basic correlation.
            'idx' is a np.ndarray or list that gives the specific locations where 'val[1]' is set.
        If correlation_structure=='rho_list', rho must be a np.ndarray or list with length dim-1.
        If correlation_structure=='block', rho is a dict with keys 'blocks' and 'val'.
            - rho['blocks'] must be tuple or list of tuples, where each tuple contains
              (correlation value of the block, start idx of the block, length of the blocks).
            - rho['val'] is the correlation value of the non-block elements.
        
    beta : float, optional
        Shape parameter.
            - If beta is equal to one, then we generate Gaussian sources.
            - If beta is less than 1, the distribution of the marginals is more peaky, with heavier
              tails.
            - If beta is greater than 1, it is less peaky with lighter tails.

    cov : np.ndarray, optional
        User-defined covariance matrix of dimensions dim x dim, where dim is the number of sources

    Returns
    -------
    X : np.ndarray
        matrix of dimensions dim x N, which contains N samples of each of the dim MGGD sources

    cov: np.ndarray
        covariance matrix of X (either generated by this function, or same as input)
        
    
    Notes
    -----
    Coded by Lionel Bombrun (lionel.bombrun at u-bordeaux.fr)
    and Zois Boukouvalas (zb1 at umbc.edu)
    Modified by Q. Long 08/2018 (qunfang1 at umbc.edu)
    Converted to Python by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    Machine Learning for Signal Processing Lab (MLSP-Lab)
    http://mlsp.umbc.edu

    Reference: E. Gomez, M. Gomez-Villegas, and J. Marin, "A multivariate generalization of the
    power exponential family of distributions," Communications in Statistics-Theory and Methods,
    vol. 27, no. 3, pp. 589{600}, 1998.

    """

    if type(N) is not int or N < 2:
        raise AssertionError("'N' must be an integer and must be at least 2.")

    if cov is None:

        if dim is None:
            raise AssertionError("Either 'dim' or 'cov' must be given.")
        elif correlation_structure is None:
            raise AssertionError("Either 'correlation_structure' or 'cov' must be given.")
        elif rho is None and correlation_structure != 'q_qt':
            raise AssertionError("Either 'rho' or 'cov' must be given.")

        if type(dim) is not int:
            raise AssertionError("'dim' must be an integer.")

        if correlation_structure == 'two_rho':
            if type(rho) is not dict:
                raise AssertionError("'rho must be dict with rho['val'] = np.ndarray, "
                                     "rho['idx'] = np.ndarray.")
            else:
                if sorted(list(rho.keys())) != ['idx', 'val']:
                    raise AssertionError("'rho' must have keys 'val' and 'idx'.")
                elif (type(rho['val']) is not np.ndarray and type(rho['val']) is not list) or (
                        type(rho['idx']) is not np.ndarray and type(rho['idx']) is not list):
                    raise AssertionError("'rho[key]' must be np.ndarray or list.")
                elif len(rho['val']) != 2:
                    raise AssertionError("'rho[val]' must consists of 2 elements.")

        elif correlation_structure == 'rho_list':
            if type(rho) is not list and type(rho) is not np.ndarray:
                raise AssertionError("'rho' must be np.ndarray or list.")

        elif correlation_structure == 'block':
            if type(rho) is not dict:
                raise AssertionError("'rho must be dict with rho['val'] = np.ndarray, "
                                     "rho['blocks'] = tuple or list of tuples.")
            else:
                if sorted(list(rho.keys())) != ['blocks', 'val']:
                    raise AssertionError("'rho' must have keys 'val' and 'blocks'.")
                elif type(rho['val']) is not float:
                    raise AssertionError("'rho[val]' must be a float value.")
                if type(rho['blocks']) is not list and type(rho['blocks']) is not tuple:
                    raise AssertionError("'rho[blocks]' must be tuple or list of tuples.")
        else:
            if type(rho) is not float and type(rho) is not int:
                raise AssertionError("'rho' must be a float or int value.")

    else:
        dim = cov.shape[0]

        if cov.ndim != 2:
            raise AssertionError("'cov' must be a 2D matrix.")
        elif cov.shape[0] != cov.shape[1]:
            raise AssertionError("cov' must be a square matrix.")
        try:
            np.linalg.cholesky(cov)
        except:
            raise AssertionError("Covariance matrix must be positive semidefinite.")

    if cov is None:
        # This is the covariance matrix of X
        cov = np.abs(_create_covariance_matrix(dim, correlation_structure, rho))

        # make sure randomly generated cov is positive semidefinite. If not, regenerate
        flag = 0
        while flag == 0:
            try:
                np.linalg.cholesky(cov)
                flag = 1
            except:
                cov = np.abs(_create_covariance_matrix(dim, correlation_structure, rho))

    u = _random_sphere(N, dim).T  # u^(n) in (3)

    # X is parametrized by scaled version of cov, see proposition 3.2 (iii)
    scaling = 2 ** (1 / beta) * gamma((dim + 2) / (2 * beta)) / (dim * gamma(dim / (2 * beta)))
    Sigma = cov / scaling

    # chol(Sigma) = A^T in (3)
    u_A = np.linalg.cholesky(Sigma) @ u

    R = np.random.gamma(dim / (2 * beta), scale=2, size=N) ** (1 / (2 * beta))  # R in (3) with density shown in (4)

    X = R * u_A

    return X, cov


def _create_covariance_matrix(dim, correlation_structure, rho):
    if correlation_structure == 'uniform':
        Sigma = np.full((dim, dim), rho)
        np.fill_diagonal(Sigma, 1)

    elif correlation_structure == 'ar':
        Sigma = np.eye(dim)
        for i in range(dim):
            for j in range(dim):
                Sigma[i, j] = rho ** (np.abs(i - j))

    elif correlation_structure == 'q_qt':
        R = normalize_row_vectors(np.random.rand(dim, dim) - 0.2)
        Sigma = R @ R.T

    elif correlation_structure == 'two_rho':
        Sigma = np.full((dim, dim), rho['val'][0])
        for i in rho['idx']:
            for j in rho['idx']:
                Sigma[i, j] = rho['val'][1]
        np.fill_diagonal(Sigma, 1)

    elif correlation_structure == 'rho_list':
        Sigma = np.eye(dim)
        n = len(rho)
        if np.mod(dim - 1, n):
            raise AssertionError("'rho' must contain 'dim'-1 elements")
        M = (dim - 1) // n
        for j in range(n):
            Sigma[0, M * j + 1:M * (j + 1) + 1] = rho[j] * np.ones(M)
            Sigma[M * j + 1:M * (j + 1) + 1, 0] = rho[j] * np.ones(M)
        for i in range(1, dim):
            for j in range(i + 1, dim):
                Sigma[i, j] = Sigma[i - 1, j]
                Sigma[j, i] = Sigma[i, j]

    elif correlation_structure == 'block':
        if type(rho['blocks']) is tuple:
            csl = [rho['blocks']]
        else:
            csl = rho['blocks']

        Sigma = np.full((dim, dim), rho['val'])
        for (correlation, start, length) in csl:
            Sigma[start:start + length, start:start + length] = correlation
        np.fill_diagonal(Sigma, 1)

    else:
        raise AssertionError("'correlation_structure' must be 'uniform', 'ar', 'q_qt', "
                             "'two_rho', 'rho_list', or 'block'.")

    return Sigma


def _random_sphere(N, dim=3):
    """
    Generates uniform random points on the surface of a unit radius N-dimensional sphere centered
    in the origin. This script uses different algorithms according to the dimensions of points:
       -2D:  random generation of theta [0 2*pi]
       -3D:  the "trig method".
       -nD:  Gaussian distribution


    Parameters
    ----------
    N : int
        Number of points to be generated

    dim : int, optional
        Dimension of data: must be at least 2


    Returns
    -------
    X : np.ndarray
        Matrix of dimensions N x dim representing the coordinates of random points generated


    Notes
    -----
    Authors: Luigi Giaccari, Ed Hoyle
    Converted to Python by Isabell Lehmann

    """

    if dim == 2:
        # just a random generation of theta
        temp = np.random.rand(N) * 2 * np.pi  # theta: use as temp value
        X = np.zeros((N, dim))
        X[:, 0] = np.cos(temp)  # x
        X[:, 1] = np.sin(temp)  # y

    elif dim == 3:
        # trig method
        X = np.zeros((N, dim))
        X[:, 2] = np.random.rand(N) * 2 - 1  # z
        t = np.random.rand(N) * 2 * np.pi
        r = np.sqrt(1 - X[:, 2] ** 2)
        X[:, 0] = r * np.cos(t)  # x
        X[:, 1] = r * np.sin(t)  # y

    else:
        # use gaussian distribution
        X = np.random.randn(N, dim)
        X /= np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]

    return X


def randmv_laplace(d, T, lambda_=1, mu=None, gamma=None):
    """

    Generate T iid samples of the d-dimensional multivariate Laplace (ML) distribution, as given in
    "On the Multivariate Laplace Distribution", by Eltoft in IEEE Sig. Proc. Letters 2006.

    Parameters
    ----------
    d : int
        dimension of the multivariate Laplacian

    T : int
        number of iid samples

    lambda_ : float
        exponential rate parameter, > 0

    mu : np.ndarray
        mean vector of dimension d

    gamma :
        internal covariance structure, note det(Gamma)=1


    Returns
    -------
    Y : np.ndarray
        ML source matrix of dimension d x_sorted T


    Notes
    -----

    Note that a method for transforming an uncorrelated ML, Y, into a correlated ML, V, is given by
    Eltoft reference using V=A@Y+b, where A is a d x d real-valued matrix and b is d-dimensional
    real-valued vector, then V is ML with parameters specified in equations (14)-(16):

    lambda_new = lambda_ * np.abs(np.linalg.det(A))**(1/d)
    mu_new = A @ mu + b
    Gamma_new = A @ Gamma @ np.conj(A.T) * np.abs(np.linalg.det(A))**(-2/d)

    The special case of mu=b=0, Gamma=eye(d), and np.linalg.det(A)=1 is nice since
    then, lambda_new=lambda, mu_new=0, and Gamma_new=A @ Gamma @ np.conj(A.T).

    Coded by Matthew Anderson 9/15/2011.
    Converted to Python by Isabell Lehmann 2021-02-01

    """

    if mu is None:
        mu = np.zeros(d)
    if gamma is None:
        gamma = np.eye(d)

    if lambda_ < 0 or np.any(np.iscomplex(lambda_)):
        raise AssertionError('Rate parameter lambda_ should be real-valued and greater than 0.')

    if mu.shape[0] != d or np.any(np.iscomplex(lambda_)):
        raise AssertionError('Mean vector should be real-valued and of dimension d.')

    if gamma.shape[0] != d or gamma.shape[1] != d or np.abs(
            np.linalg.det(gamma) - 1) > 0.0001 or np.any(np.iscomplex(gamma)):
        raise AssertionError(
            'Internal covariance structure needs to be a real-valued square matrix with a determinant of one.')

    X = np.random.randn(d, T)
    Z = np.sqrt(np.random.exponential(1 / lambda_, T))
    Y = mu[:, np.newaxis] + Z * (scipy.linalg.sqrtm(gamma) @ X)  # equation (6)

    return Y
