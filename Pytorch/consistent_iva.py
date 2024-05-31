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

# parallel processing
import multiprocessing
from joblib import Parallel, delayed

from tqdm import tqdm

from .iva_g import iva_g
from .iva_l_sos import iva_l_sos
from .helpers_iva import _bss_isi


def consistent_iva(X, which_iva='iva_g', n_runs=20, parallel=True, **kwargs):
    """
    IVA is performed n_runs times with different initalizations.
    The most consistent demixing matrix along with the change in W for each iteration is returned,
    with the corresponding sources, mixing matrix, SCV covariance matrices.
    Consistence is measured by cross joint ISI, which can be returned optionally.


    Parameters
    ----------
    X : np.ndarray
        data matrix of dimensions N x T x K (sources x samples x datasets)

    which_iva : str, optional
        'iva_g' or 'iva_l_sos'

    n_runs : int, optional
        how many times iva is performed

    kwargs : list
        keyword arguments for the iva function


    Returns
    -------
    W : np.ndarray
        demixing matrix of dimensions N x N x K

    W_change : list
        change in W for each iteration

    cross_isi_jnt : np.ndarray
        cross joint isi for each run. Only returned if return_cross_isi==True.


    Returns
    -------
    iva_results : dict
        - 'W' : estimated demixing matrix of most consistent run of dimensions N x N x K
        - 'W_change' : change in W for each iteration of most consistent run
        - 'S' : estimated sources of dimensions N x T x K
        - 'A' : estimated mixing matrix of dimensions N x N x K
        - 'scv_cov' : covariance matrices of the SCVs, of dimensions K x K x N
        - 'cross_isi' : cross joint isi for each run
        - 'cost' : cost of each run
        - 'isi' : isi for each run if true A is supplied, else None


    Notes
    -----
    Code written by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    Reference:
    Long, Q., C. Jia, Z. Boukouvalas, B. Gabrielson, D. Emge, and T. Adali.
    "Consistent run selection for independent component analysis: Application
    to fMRI analysis." IEEE International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2018.
    """

    # calculate demixing matrices for several runs
    if parallel:
        if which_iva == 'iva_g':
            W, cost, _, isi, W_change = zip(*Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(iva_g)(X, return_W_change=True, **kwargs) for i in tqdm(range(n_runs))))
        elif which_iva == 'iva_l_sos':
            W, cost, _, isi, W_change = zip(*Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(iva_l_sos)(X, return_W_change=True, **kwargs) for i in tqdm(range(n_runs))))
        else:
            raise AssertionError("which_iva must be 'iva_g' or 'iva_l_sos'")
        cost = [c[-1] for c in cost]
        if isi[0] is not None:
            isi = [i[-1] for i in isi]

    else:
        W = []
        cost = []
        isi = []
        W_change = []
        for run in tqdm(range(n_runs)):
            if which_iva == 'iva_g':
                temp = iva_g(X, return_W_change=True, **kwargs)
            elif which_iva == 'iva_l_sos':
                temp = iva_l_sos(X, return_W_change=True, **kwargs)
            else:
                raise AssertionError("which_iva must be 'iva_g' or 'iva_l_sos'")
            W.append(temp[0])
            cost.append(temp[1])
            isi.append(temp[3])
            W_change.append(temp[4])

        cost = [c[-1] for c in cost]
        if isi[0] is not None:
            isi = [i[-1] for i in isi]

    # use cross joint isi to find most consistent run
    selected_run, _, cross_jnt_isi, _ = _run_selection_cross_isi(W)

    W = W[selected_run]
    W_change = W_change[selected_run]

    # get dimensions
    N, T, K = X.shape

    S = np.zeros((N, T, K))
    for k in range(K):
        S[:, :, k] = W[:, :, k] @ X[:, :, k]

    A_hat = np.zeros((N, N, K))
    for k in range(K):
        A_hat[:, :, k] = np.linalg.lstsq(S[:, :, k].T, X[:, :, k].T)[0].T

    scv = np.zeros((K, T, N))
    for n in range(N):
        for k in range(K):
            scv[k, :, n] = S[n, :, k]

    scv_cov = np.zeros((K, K, N))
    for n in range(N):
        scv_cov[:, :, n] = np.cov(scv[:, :, n])

    # results
    iva_results = {'W': W, 'S': S, 'A': A_hat, 'scv_cov': scv_cov, 'W_change': W_change,
                   'cross_isi': cross_jnt_isi, 'cost': cost, 'isi': isi}

    return iva_results


def _run_selection_cross_isi(W):
    """
    For a given list of estimated demixing matrices, return the index which achieves the
    smallest cross joint isi and the index which achieves the smallest cross average isi.


    Parameters
    ----------
    W : list
        estimated demixing matrix for several runs, each matrix is of dimensions N x N x K
        (N: #components, K: #datasets)


    Returns
    -------
    selected_run_jnt : int
        idx of the run that achieves minimum cross joint isi

    selected_run_avg : int
        idx of the run that achieves minimum cross average isi

    cross_isi_jnt : np.ndarray
        cross joint isi for each run

    cross_isi_avg : np.ndarray
        cross average isi for each run


    Notes
    -----
    Code written by Isabell Lehmann (isabell.lehmann at sst.upb.de)

    Reference:
    Long, Q., C. Jia, Z. Boukouvalas, B. Gabrielson, D. Emge, and T. Adali.
    "Consistent run selection for independent component analysis: Application
    to fMRI analysis." IEEE International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2018.
    """

    n_runs = len(W)
    cross_isi_avg_matrix = np.zeros((n_runs, n_runs))
    cross_isi_jnt_matrix = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        W_inv = np.zeros_like(W[i])
        for k in range(W_inv.shape[2]):
            W_inv[:, :, k] = np.linalg.inv(W[i][:, :, k])
        for j in range(n_runs):
            cross_isi_avg_matrix[i, j], cross_isi_jnt_matrix[i, j] = _bss_isi(W=W[j], A=W_inv)

    # find most consistent run
    cross_isi_jnt = np.sum(cross_isi_jnt_matrix, axis=1) / (n_runs - 1)
    cross_isi_avg = np.sum(cross_isi_avg_matrix, axis=1) / (n_runs - 1)
    selected_run_jnt = np.argmin(cross_isi_jnt)
    selected_run_avg = np.argmin(cross_isi_avg)

    return selected_run_jnt, selected_run_avg, cross_isi_jnt, cross_isi_avg
