import numpy as np
import matplotlib.pyplot as plt

from ..iva_g import iva_g


def test_iva_g():
    """
    test iva_second_order
    """

    N = 10  # number of sources
    T = 1000  # sample size
    K = 10  # number of groups

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

    # separation
    W, _, _, isi = iva_g(X, A=A, jdiag_initW=True)
    np.testing.assert_array_less(isi, 0.05)

    # show results
    T1 = np.zeros((N, N))
    for k in range(K):
        T_k = W[:, :, k] @ A[:, :, k]
        T_k = np.abs(T_k)
        for n in range(N):
            T_k[n, :] /= np.amax(np.abs(T_k[n, :]))
        T1 += T_k / K

    P = np.zeros((N, N))
    imax = np.argmax(T1, axis=0)
    P[np.arange(N), imax] = 1
    T1 = P @ T1

    plt.figure()
    plt.imshow(T1, extent=[0, N, 0, N], cmap='bone')
    plt.title('joint global matrix')
    plt.colorbar()
    plt.show()

    print('Ideally, image is identity matrix.')


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
    W, _, _, isi = iva_g(X, verbose=True, A=A, jdiag_initW=True, max_iter=4)

    # CCA init (for 2 datasets)
    W, _, _, isi = iva_g(X[:, :, 0:2], verbose=True, A=A[:, :, 0:2], jdiag_initW=True, max_iter=4)

    # W_init is given
    W_init = np.zeros_like(A)
    for k in range(K):
        W_init[:, :, k] = np.linalg.inv(A[:, :, k]) + np.random.randn(N, N) * 0.1

    W, _, _, isi = iva_g(X, verbose=True, A=A, W_init=W_init, max_iter=4)

    # same W_init for each dataset
    W, _, _, isi = iva_g(X, verbose=True, A=A, W_init=W_init[:, :, 0], max_iter=4)

    # random init
    W, _, _, isi = iva_g(X, verbose=True, A=A, max_iter=4)

    # gradient optimization approach
    W, _, _, isi = iva_g(X, opt_approach='gradient', verbose=True, A=A, max_iter=4)

    # quasi optimization approach
    W, _, _, isi = iva_g(X, opt_approach='quasi', verbose=True, A=A, max_iter=4)

    # use int step size
    W, _, _, isi = iva_g(X, verbose=True, A=A, max_iter=4, alpha0=2)

    # complex call with real-valued data
    W, _, _, isi = iva_g(X, complex_valued=True, verbose=True, A=A, max_iter=4)

    # circular call with real-valued data
    W, _, _, isi = iva_g(X, verbose=True, circular=True, A=A, max_iter=4)

    # No whitening
    W, _, _, isi = iva_g(X, whiten=False, verbose=True, A=A, max_iter=4)


def test_iva_complex_function_calls():
    """
    Make sure that function calls do not raise errors. Final value is not that important, therefore
    max-iter is set to 4.
    """

    N = 20  # number of sources
    T = 1000  # sample size
    K = 40  # number of groups

    # generate the mixtures
    S = np.zeros((N, T, K), dtype=complex)
    for n in range(N):
        temp1 = np.random.randn(K, T) + 1j * np.random.randn(K, T)
        temp = np.zeros((K, T), dtype=complex)
        B = np.random.randn(K, K, 3) + 1j * np.random.randn(K, K, 3)

        for p in range(2):
            for t in range(2, T):
                # introduce nonwhiteness and spatial correlation
                temp[:, t] += B[:, :, p] @ np.conj(temp1[:, t - p])

        for k in range(K):
            S[n, :, k] = temp[k, :]
            S[n, :, k] -= np.mean(S[n, :, k])
            S[n, :, k] = S[n, :, k] / np.std(S[n, :, k], ddof=1)

    A = np.random.randn(N, N, K) + 1j * np.random.randn(N, N, K)
    X = np.zeros((N, T, K), dtype=complex)
    for k in range(K):
        X[:, :, k] = A[:, :, k] @ S[:, :, k]

    # initialize with multi-set diagonalization
    W, _, _, isi = iva_g(X, verbose=True, A=A, jdiag_initW=True, max_iter=4)

    # CCA init (for 2 datasets)
    W, _, _, isi = iva_g(X[:, :, 0:2], verbose=True, A=A[:, :, 0:2], jdiag_initW=True, max_iter=4)

    # W_init is given
    W_init = np.zeros_like(A)
    for k in range(K):
        W_init[:, :, k] = np.linalg.inv(A[:, :, k]) + \
                          np.random.randn(N, N) * 0.1 + 1j * np.random.randn(N, N) * 0.1

    W, _, _, isi = iva_g(X, verbose=True, A=A, W_init=W_init, max_iter=4)

    # same W_init for each dataset
    W, _, _, isi = iva_g(X, verbose=True, A=A, W_init=W_init[:, :, 0], max_iter=4)

    # random init
    W, _, _, isi = iva_g(X, verbose=True, A=A, max_iter=4)

    # gradient optimization approach
    W, _, _, isi = iva_g(X, opt_approach='gradient', verbose=True, A=A, max_iter=4)

    # quasi optimization approach
    W, _, _, isi = iva_g(X, opt_approach='quasi', verbose=True, A=A, max_iter=4)

    # use int step size
    W, _, _, isi = iva_g(X, verbose=True, A=A, max_iter=4, alpha0=2)

    # circular

    W, _, _, isi = iva_g(X, circular=True, verbose=True, A=A, max_iter=4)

    W, _, _, isi = iva_g(X, opt_approach='gradient', circular=True, verbose=True, A=A, max_iter=4)

    W, _, _, isi = iva_g(X, opt_approach='quasi', circular=True, verbose=True, A=A, max_iter=4)
