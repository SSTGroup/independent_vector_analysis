import numpy as np

from ..helpers_iva import _normalize_column_vectors, _bss_isi, whiten_data


def test_normalize_column_vectors():
    A = np.random.rand(10, 20) * 3
    A_n = _normalize_column_vectors(A)
    np.testing.assert_array_almost_equal(np.linalg.norm(A_n, axis=0), np.ones(20))


def test_bss_isi():
    N = 10  # number of sources

    # generate orthogonal matrices for 1 dataset
    A = np.random.randn(N, N)
    U, _, Vh = np.linalg.svd(A)
    A = U @ Vh
    B = np.random.randn(N, N)
    U, _, Vh = np.linalg.svd(B)
    B = U @ Vh

    np.testing.assert_array_less(_bss_isi(A, B)[0], 0.7)
    np.testing.assert_almost_equal(_bss_isi(A, A.T)[0], 0)

    # generate orthogonal matrices for K datasets
    K = 2  # number of datasets
    A = np.random.randn(N, N, K)
    for k in range(K):
        U, _, Vh = np.linalg.svd(A[:, :, k])
        A[:, :, k] = U @ Vh
    B = np.random.randn(N, N, K)
    for k in range(K):
        U, _, Vh = np.linalg.svd(B[:, :, k])
        B[:, :, k] = U @ Vh

    np.testing.assert_array_less(_bss_isi(A, B)[1], 0.7)
    np.testing.assert_almost_equal(_bss_isi(A, np.moveaxis(A, [0, 1, 2], [1, 0, 2]))[1], 0)


def test_whiten_data():
    N = 10  # number of sources
    T = 1000  # sample size
    K = 2  # number of groups

    # Gaussian distributed with mean 1 and std 3
    X = np.random.randn(N, T, K) * 3 + 1 + 1j * (np.random.randn(N, T, K) * 3 + 1)

    # for 3D matrix x
    Z, V = whiten_data(X)
    for k in range(K):
        ZZh = np.cov(Z[:, :, k], ddof=0)
        np.testing.assert_array_almost_equal(np.real(ZZh), np.eye(N))
        np.testing.assert_array_almost_equal(np.imag(ZZh), np.zeros((N, N)))

    # for 2D matrix x
    Z, V = whiten_data(X[:, :, 0])
    ZZh = np.cov(Z, ddof=0)
    np.testing.assert_array_almost_equal(np.real(ZZh), np.eye(N))
    np.testing.assert_array_almost_equal(np.imag(ZZh), np.zeros((N, N)))

    # with dimension reduction
    reduced_dim = 6

    # for 3D matrix x
    Z, V = whiten_data(X, reduced_dim)
    for k in range(K):
        ZZh = np.cov(Z[:, :, k], ddof=0)
        np.testing.assert_array_almost_equal(np.real(ZZh), np.eye(reduced_dim))
        np.testing.assert_array_almost_equal(np.imag(ZZh), np.zeros((reduced_dim, reduced_dim)))

    # for 2D matrix x
    Z, V = whiten_data(X[:, :, 0], reduced_dim)
    ZZh = np.cov(Z, ddof=0)
    np.testing.assert_array_almost_equal(np.real(ZZh), np.eye(reduced_dim))
    np.testing.assert_array_almost_equal(np.imag(ZZh), np.zeros((reduced_dim, reduced_dim)))
