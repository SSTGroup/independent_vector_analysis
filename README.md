# Independent Vector Analysis
   
This package contains the Python versions of IVA-G and IVA-L-SOS, converted from the [MLSP-Lab MATLAB Codes](http://mlsp.umbc.edu/resources.html).

- **Website:** http://mlsp.umbc.edu/jointBSS_introduction.html
- **Source-code:** https://github.com/SSTGroup/independent_vector_analysis


## Installing independent_vector_analysis

The only pre-requisite is to have **Python 3** (>= version 3.6) installed.
The iva package can be installed with

    pip install independent_vector_analysis

Required third party packages will automatically be installed.


## Quickstart

First, the imports:

    import numpy as np
    from independent_vector_analysis import iva_g, consistent_iva
    from independent_vector_analysis.data_generation import MGGD_generation

Create a dataset with N=3 sources, which are correlated across K=4 datasets.
Each source consists of T=10000 samples:
    
    N = 3
    K = 4
    T = 10000
    rho = 0.7
    S = np.zeros((N, T, K))
    for idx in range(N):
        S[idx, :, :] = MGGD_generation(T, K, 'ar', rho, 1)[0].T
    A = np.random.randn(N,N,K)
    X = np.einsum('MNK, NTK -> MTK', A, S)
    W, cost, Sigma_n, isi = iva_g(X, A=A, jdiag_initW=False)

Apply IVA-G to reconstruct the sources.
If the mixing matrix *A* is passed, the ISI is calculated.
Let the demixing matrix W be initialized by joint diagonalization:

    W, cost, Sigma_n, isi = iva_g(X, A=A, jdiag_initW=False)

*W* is the estimated demixing matrix.
*cost* is the cost for each iteration.
*Sigma_n*[:,:,n] contains the covariance matrix of the nth SCV.
*isi* is the joint ISI for each iteration.

Find the most consistent result of 500 runs in IVA-L-SOS:
    
    iva_results = consistent_iva(X, which_iva='iva_l_sos', n_runs=500)

where *iva_results* is a dict containing:
* 'W' : estimated demixing matrix of dimensions N x N x K
* 'W_change' : change in W for each iteration
* 'S' : estimated sources of dimensions N x T x K
* 'A' : estimated mixing matrix of dimensions N x N x K
* 'scv_cov' : covariance matrices of the SCVs, of dimensions K x K x N (the same as *Sigma_n* in iva_g / iva_l_sos)
* 'cross_isi' : cross joint ISI for each run compated with all other runs

[comment]: <> (If you see a bug, open an [issue]&#40;https://github.com/tensorly/tensorly/issues&#41;, or better yet, a [pull-request]&#40;https://github.com/tensorly/tensorly/pulls>&#41;!)

## Contact

In case of questions, suggestions, problems etc. please send an email to isabell.lehmann@sst.upb.de, or open an issue here on Github.

## Citing

If you use this package in an academic paper, please cite [1]

    @inproceedings{Lehmann2022,
      author  = {Lehmann, Isabell and Acar, Evrim and Hasija, Tanuj and Akhonda, M.A.B.S. and Calhoun, Vince D. and Schreier, Peter J. and Adali, Tulay},
      title   = {Multi-task fMRI Data Fusion Using IVA and PARAFAC2},
      journal = {2022 IEEE International Conference on Acoustics, Speech and Signal Processing},
	  year = {2022}
    }
    
    
[1] Isabell Lehmann, Evrim Acar, et al., **Multi-task fMRI Data Fusion Using IVA and PARAFAC2**, *2022 IEEE International Conference on Acoustics, Speech and Signal Processing*, 2022.



