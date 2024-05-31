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

import torch
import numpy as np
import scipy as sc
import scipy.linalg
from scipy.special import gamma



def _normalize_column_vectors(x):
    return x / x.norm(dim=0)

def _decouple_trick(W, n, Q=None, R=None):
    """
    Computes the H vector for the decoupling trick [1] of the nth row of W.
    There are many possible methods for computing H. This algorithm just implements the method in
    [1] (in the MATLAB version, there are more methods implemented): A recursive QR algorithm is
    used to compute H.


    Parameters
    ----------
    W : np.ndarray
        stacked demixing matrix of dimensions N x N x K

    n : int
        row index of W for that H vector is calculated

    Q : np.ndarray, optional
        stacked matrix of dimensions N x N x K. Must be provided if n > 0

    R : np.ndarray, optional
        stacked matrix of dimensions N x N-1 x K. Must be provided if n > 0


    Returns
    -------
    H : np.ndarray
        H vector (dimensions N x K) for the decoupling trick [1] of the nth row of W
    Q_ : np.ndarray, optional
        stacked matrix of dimensions N x N x K
    R_ : np.ndarray, optional
        stacked matrix of dimensions N x N-1 x K


    Notes
    -----
    Main References:
    [1] X.-L. Li & X.-D. Zhang, "Nonorthogonal Joint Diagonalization Free of Degenerate Solution,"
    IEEE Trans. Signal Process., 2007, 55, 1803-1814

    Coded by Matthew Anderson (matt.anderson@umbc.edu)
    Converted to Python by Isabell Lehmann (isabell.lehmann@sst.upb.de)

    Version 01 - 20120919 - Initial publication
    Version 02 - 20210129 - Converted to Python

    """

    # get dimensions
    M, N, K = W.shape
    if M != N:
        raise AssertionError('Assuming W is square matrix.')
    H = np.zeros((N, K), dtype=W.dtype, device=W.device)

    # use QR recursive method
    if n == 0:
        Q_ = torch.zeros((N, N, K), dtype=W.dtype, device=W.device)
        R_ = torch.zeros((N, N - 1, K), dtype=W.dtype, device=W.device)
    else:
        Q_ = Q.clone()
        R_ = R.clone()

    for k in range(K):
        if n == 0:
            W_tilde = W[1:N, :, k]
            Q_[:, :, k], R_[:, :, k] = torch.qr(W_tilde.conj().t(), some=True)
        else:
            n_last = n - 1
            e_last = torch.zeros(N - 1, dtype=W.dtype, device=W.device)
            e_last[n_last] = 1

            # e_last.shape and R.shape[1], W.shape[1] and Q.shape[1] must be equal
            Q_[:, :, k], R_[:, :, k] = sc.qr_update(Q_[:, :, k], R_[:, :, k],
                                                    -W[n, :, k].conj(), e_last)
            Q_[:, :, k], R_[:, :, k] = sc.qr_update(Q_[:, :, k], R_[:, :, k],
                                                    W[n_last, :, k].conj(), e_last)

        H[:, k] = Q_[:, -1, k]

    return H, Q_, R_



def _bss_isi(W, A, s=None):
    """
    Calculate measure of quality of separation for blind source separation algorithms.
    Model:   x = A @ s,   y = W @ x = W @ A @ s

    Parameters
    ----------
    W : torch.Tensor
        demixing matrix of dimensions N x N x K or M x p

    A : torch.Tensor
        true mixing matrix of dimension N x N x K or p x N

    s : torch.Tensor, optional
        true sources of dimension N x T x K or N x T

    (p: #sensors, N: #sources, M: #estimatedsources, K: #datasets, T: #samples)

    Returns
    -------
    avg_isi : float
        avg_isi=0 is optimal.
        Normalized performance index is given in Choi, S. Cichocki, A. Zhang, L. & Amari, S.
        Approximate maximum likelihood source separation using the natural gradient
        Wireless Communications, 2001. (SPAWC '01). 2001 IEEE Third Workshop on Signal
        Processing Advances in, 2001, 235-238.

    joint_isi : float, optional
        joint_isi = 0 is optimal. Only calculated if there are at least 2 datasets, otherwise
        np.nan.
        Normalized joint performance index given in Anderson, Matthew, Tuelay Adali, and
        Xi-Lin Li. "Joint blind source separation with multivariate Gaussian model: Algorithms and
        performance analysis." IEEE Transactions on Signal Processing 60.4 (2011): 1672-1683.

    Notes
    -----
    W is the estimated demixing matrix and A is the true mixing matrix.  It should be noted
    that rows of the mixing matrix should be scaled by the necessary constants such that each
    source has unit variance and accordingly each row of the demixing matrix should be
    scaled such that each estimated source has unit variance.

    Note that A is p x N, where p is the number of sensors and N is the number of signals
    and W is M x p, where M is the number of estimated signals.  Ideally M=N but this is not
    guaranteed.  So if M > N, the algorithm has estimated more sources than it "should", and
    if M < N the algorithm has not found all of the sources.

    Code is converted to PyTorch by [Your Name]
    """

    # generalized permutation invariant flag (default=False), only used when s is None
    gen_perm_inv_flag = False

    if W.dim() == 2 and A.dim() == 2:
        if s is None:
            # Traditional metric, user provided W & A separately
            G = torch.matmul(W, A)
            M, N = G.shape
            G_abs = torch.abs(G)
            if gen_perm_inv_flag:
                # normalization by row
                G_abs /= torch.max(G_abs, dim=1, keepdim=True)[0]
        else:
            # Equalize energy associated with each estimated source and true source.
            y = torch.matmul(torch.matmul(W, A), s)
            # Standard deviation calculated with n-1
            D = torch.diag(
                1 / torch.std(s, dim=1, unbiased=True))  # s_norm = D @ s, where s_norm has unit variance
            U = torch.diag(
                1 / torch.std(y, dim=1, unbiased=True))  # y_norm = U @ y, where y_norm has unit variance

            # Thus: y_norm = U @ W @ A @ np.linalg.inv(D) @ s_norm = G @ s_norm, and
            G = torch.matmul(torch.matmul(U, W), torch.solve(A.t(), D.t())[0].t())  # A @ np.linalg.inv(D)
            M, N = G.shape
            G_abs = torch.abs(G)

        avg_isi = 0
        for m in range(M):
            avg_isi += torch.sum(G_abs[m, :]) / torch.max(G_abs[m, :]) - 1

        for n in range(N):
            avg_isi += torch.sum(G_abs[:, n]) / torch.max(G_abs[:, n]) - 1

        avg_isi /= (2 * N * (N - 1))

        return avg_isi, torch.nan

    elif W.dim() == 3 and A.dim() == 3:
        # IVA/GroupICA/MCCA Metrics
        # For this we want to average over the K groups as well as provide the additional
        # measure of solution to local permutation ambiguity (achieved by averaging the K
        # demixing-mixing matrices and then computing the ISI of this matrix).

        N, M, K = W.shape
        if M != N:
            raise AssertionError('This more general case has not been considered here.')

        avg_isi = 0
        G_abs_total = torch.zeros((N, N))
        G = torch.zeros((N, N, K), dtype=W.dtype)
        for k in range(K):
            if s is None:
                # Traditional metric, user provided W & A separately
                G_k = torch.matmul(W[:, :, k], A[:, :, k])
                G_abs = torch.abs(G_k)
                if gen_perm_inv_flag:
                    # normalization by row
                    G_abs /= torch.max(G_abs, dim=1, keepdim=True)[0]
            else:

                # Equalize energy associated with each estimated source and true source.
                # Standard deviation calculated with n-1
                y_k = torch.matmul(torch.matmul(W[:, :, k], A[:, :, k]), s[:, :, k])
                D_k = torch.diag(1 / torch.std(s[:, :, k], dim=1,
                                               unbiased=True))  # s_norm = D @ s, where s_norm has unit variance
                U_k = torch.diag(1 / torch.std(y_k, dim=1,
                                               unbiased=True))  # y_norm = U @ y, where y_norm has unit variance
                # Thus: y_norm = U @ W @ A @ np.linalg.inv(D) @ s_norm = G @ s_norm, and
                G_k = torch.matmul(torch.matmul(U_k, W[:, :, k]), torch.solve(A[:, :, k].t(), D_k.t())[0].t())  # A @ np.linalg.inv(D)
                G_abs = torch.abs(G_k)

            G[:, :, k] = G_k

            G_abs_total += G_abs

            for n in range(N):
                avg_isi += torch.sum(G_abs[n, :]) / torch.max(G_abs[n, :]) - 1

            for m in range(N):
                avg_isi += torch.sum(G_abs[:, m]) / torch.max(G_abs[:, m]) - 1

        avg_isi /= (2 * N * (N - 1) * K)

        G_abs = torch.copy(G_abs_total)
        if gen_perm_inv_flag:
            # normalization by row
            G_abs /= torch.max(G_abs, dim=1, keepdim=True)[0]

        joint_isi = 0
        for n in range(N):
            joint_isi += torch.sum(G_abs[n, :]) / torch.max(G_abs[n, :]) - 1

        for m in range(N):
            joint_isi += torch.sum(G_abs[:, m]) / torch.max(G_abs[:, m]) - 1

        joint_isi /= (2 * N * (N - 1))

        return avg_isi, joint_isi
    else:
        raise AssertionError('All inputs must be of either dimension 2 or 3')


def whiten_data(x, dim_red=None):
    if dim_red is None:
        dim_red = x.shape[0]

    if x.ndim == 2:
        N, T = x.shape

        # Step 1. Center the data.
        x_zm = x - torch.mean(x, dim=1, keepdim=True)

        # Step 2. Form MLE of data covariance.
        covar = torch.matmul(x_zm, x_zm.transpose(0, 1)) / T

        # Step 3. Eigen decomposition of covariance.
        eigval, eigvec = torch.linalg.eigh(covar)

        # sort eigenvalues and corresponding eigenvectors in descending order
        eigval = torch.flip(eigval, dims=[0])
        eigvec = torch.fliplr(eigvec)

        # Step 4. Forming whitening transformation.
        V = torch.einsum('n,Nn -> nN', 1 / torch.sqrt(eigval[0:dim_red]), torch.conj(eigvec[:, 0:dim_red]))

        # Step 5. Form whitened data
        z = torch.matmul(V, x_zm)

    else:
        N, T, K = x.shape

        eigval = torch.zeros((N, K), dtype=x.dtype)
        eigvec = torch.zeros((N, N, K), dtype=x.dtype)
        x_zm = torch.zeros_like(x)

        for k in range(K):
            # Step 1. Center the data.
            x_zm[:, :, k] = x[:, :, k] - torch.mean(x[:, :, k], dim=1, keepdim=True)

            # Step 2. Form MLE of data covariance.
            covar = torch.matmul(x_zm[:, :, k], x_zm[:, :, k].transpose(0, 1)) / T

            # Step 3. Eigen decomposition of covariance.
            eigval[:, k], eigvec[:, :, k] = torch.linalg.eigh(covar)

        # sort eigenvalues and corresponding eigenvectors in descending order
        eigval = torch.flipud(eigval)
        eigvec = torch.flip(eigvec, dims=[1])

        # Step 4. Forming whitening transformation.
        V = torch.einsum('nk,Nnk -> nNk', 1 / torch.sqrt(eigval[0:dim_red, :]),
                      torch.conj(eigvec[:, 0:dim_red, :]))

        # Step 5. Form whitened data
        z = torch.einsum('nNk, Nvk-> nvk', V, x_zm)
    return z, V



def _comp_l_sos_cost(W, Y, const_log=None, scale_sources=False):
    N, T, K = Y.shape
    cost = 0
    xi_shape = K + 1  # .. Modified by Suchita Bhinge... September 14 2018
    if const_log is None:
        const_Ck = 0.5 * gamma(K / 2) * xi_shape ** (K / 2) / (torch.pi ** (K / 2) * gamma(K))
        const_log = -torch.log(const_Ck)

    if scale_sources:
        for k in range(K):
            ypower = torch.diag(1 / T * torch.matmul(Y[:, :, k], torch.conj(Y[:, :, k].t())))
            W[:, :, k] /= torch.sqrt(ypower)[:, None]

    for k in range(K):
        cost -= torch.log(torch.abs(sc.det(W[:, :, k])))

    for n in range(N):
        yn = Y[n, :, :].t()  # K x T
        CovN = 1 / T * torch.matmul(yn, torch.conj(yn.t()))
        gip = torch.sum(torch.conj(yn) * sc.solve(CovN, yn), dim=0)
        dcost = (const_log + 0.5 * torch.log(sc.det(CovN))) + xi_shape ** 0.5 * torch.mean(
            gip ** 0.5)  # .. Modified by Suchita Bhinge... September 14 2018
        cost += dcost
    y = Y.clone()

    return cost, y


def _resort_scvs(W, R_xx, whiten=False, V=None, complex_valued=False, circular=False, P_xx=None):
    """
    Resort order of SCVs: Order the components from most to least ill-conditioned
    """

    N, _, K = W.shape

    # First, compute the data covariance matrices (by undoing any whitening)
    if whiten:
        for k1 in range(K):
            for k2 in range(k1, K):
                R_xx[:, :, k1, k2] = torch.solve(R_xx[:, :, k1, k2], V[:, :, k2])[0].T @ torch.solve(R_xx[:, :, k1, k2].T, V[:, :, k1])[0]
                R_xx[:, :, k2, k1] = R_xx[:, :, k1, k2].conj().T  # R_xx is Hermitian

    # Second, compute the determinant of the SCVs
    if complex_valued:
        detSCV = torch.zeros(N, dtype=torch.complex64)
        Sigma_N = torch.zeros((K, K, N), dtype=torch.complex64)
    else:
        detSCV = torch.zeros(N)
        Sigma_N = torch.zeros((K, K, N))

    for n in range(N):
        # Efficient version of Sigma_n = 1/T * Y_n @ np.conj(Y_n.T) with Y_n = W_n @ X_n
        if complex_valued:
            Sigma_n = torch.zeros((K, K), dtype=torch.complex64)
        else:
            Sigma_n = torch.zeros((K, K))

        for k1 in range(K):
            for k2 in range(k1, K):
                Sigma_n[k1, k2] = W[n, :, k1] @ R_xx[:, :, k1, k2] @ W[n, :, k2].conj()
                Sigma_n[k2, k1] = Sigma_n[k1, k2].conj()  # sigma_n is Hermitian
        Sigma_N[:, :, n] = Sigma_n

        if complex_valued and not circular:
            Sigma_P_n = torch.zeros((K, K), dtype=torch.complex64)  # pseudo = 1/T * Y_n @ Y_n.T
            for k1 in range(K):
                for k2 in range(k1, K):
                    Sigma_P_n[k1, k2] = W[n, :, k1] @ P_xx[:, :, k1, k2] @ W[n, :, k2]
                    Sigma_P_n[k2, k1] = Sigma_P_n[k1, k2]  # sigma_P_n is symmetric
            detSCV[n] = torch.det(torch.cat([torch.cat([Sigma_n, Sigma_P_n], dim=-1),
                                             torch.cat([Sigma_P_n.conj().t(), Sigma_n.conj()], dim=-1)], dim=-2))
        else:
            detSCV[n] = torch.det(Sigma_n)

    # Third, sort and apply
    isort = torch.argsort(detSCV)
    Sigma_N = Sigma_N[:, :, isort]
    for k in range(K):
        W[:, :, k] = W[isort, :, k]

    return W, Sigma_N