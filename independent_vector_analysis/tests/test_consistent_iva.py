import numpy as np

from ..consistent_iva import consistent_iva

import time


def test_consistent_iva_calls():
    """
    Make sure that function calls do not raise errors. Final value is not that important, therefore
    max-iter is set to 20.
    """

    N = 20  # number of sources
    T = 500  # sample size
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

    start = time.time()
    iva_results = consistent_iva(X, which_iva='iva_g', n_runs=20, parallel=True, max_iter=20)
    end = time.time()
    print(f'Parallel loop: {end - start} seconds')

    start = time.time()
    iva_results = consistent_iva(X, which_iva='iva_g', n_runs=20, parallel=False, max_iter=20)
    end = time.time()
    print(f'Sequential loop: {end - start} seconds')
