import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from iva_g import iva_g
from helpers_iva import whiten_data
from concurrent.futures import ThreadPoolExecutor
import cProfile


def sym(A):
    if A.ndim == 2:
        return (A.T+A)/2
    else:
        return (A + np.moveaxis(A,0,1))/2

def cov_X(X):
    _,T,_ = X.shape
    return np.einsum('NTK,MTJ->KJNM',X,X)/T

def spectral_norm(M):
    if M.ndim == 2:
        return np.linalg.norm(M,ord=2)
    else:
        return np.max(np.linalg.norm(M,ord=2,axis=(0,1)))
    
def spectral_norm_extracted(Rx,K,N):
    return np.max(np.linalg.norm(np.reshape(np.moveaxis(Rx,1,0),(K,K*N,N)),ord=2,axis=(1,2)))

def smallest_singular_value(C):
    _,s,_ = np.linalg.svd(np.moveaxis(C,2,0))
    return np.min(s)

def blocks_to_full(W_blocks,K,N):
    W_full = np.zeros((N,N,K))
    for k,W_k in enumerate(W_blocks):
        begin = 0
        for W_kl in W_k:
            n_l = np.size(W_kl)
            W_full[begin:begin+n_l,begin:begin+n_l,k] = W_kl
            begin += n_l
    return W_full

def full_to_blocks( W_full,idx_W,K):
    W_blocks = []
    for k in range(K):
        W_k = []
        L_k = len(idx_W[k])
        for l in range(L_k):
            W_kl = W_full[idx_W[k][l],idx_W[k][l],k]
            W_k.append(W_kl)
        W_blocks.append(W_k)
    return W_blocks

# def quick_block_diag(W):
#     N, N, K = W.shape
#     W_bd = np.zeros((K, K * N, N))

#     def fill_block(k):
#         nonlocal W_bd
#         W_bd[k, k * N:(k + 1) * N, :] = W[:, :, k].T

#     # Utilisation d'un ThreadPoolExecutor pour paralléliser les boucles
#     with ThreadPoolExecutor() as executor:
#         executor.map(fill_block, range(K))

#     return W_bd

def lipschitz(C,lam):
    return spectral_norm(C)*lam

def joint_isi(W,A):
    N,_,_ = W.shape
    G_bar = np.sum(np.abs(np.einsum('nNk, Nvk -> nvk',W,A)),axis=2)
    score = (np.sum(np.sum(G_bar/np.max(G_bar,axis=0),axis=0)-1) + np.sum(np.sum(G_bar.T/np.max(G_bar.T,axis=0),axis=0)-1))
    return score/(2*N*(N-1))

def decrease(cost,verbose=0):
    accr = np.array(cost)[:-1] - np.array(cost)[1:]
    if np.all(accr >= 0):
        return True
    else:
        if verbose >= 1:
            for i in range(len(accr)):
                if accr[i] < 0:
                    print("increase at index :",i)
                    print("an increase of :", -accr[i])
                    break
        return False
    
def diff_criteria(A,B,mode='full'):
    
# calculates the distance between two tensors or matrices A and B taking into account the scaling ambiguity
    N = A.shape[0]
    if mode == 'full':
        if A.shape != B.shape:
            raise("A and B must be of the same dimension")
        elif A.ndim < 2 or A.ndim > 3:
            raise("Only tensors of order 2 or 3 are accepted")
        D = A-B
        return np.max(np.sum(D**2, axis=1))/(2*N)
    # elif mode =='blocks':
    #     # A et B sont des listes de K matrices, 
    #     # elles mêmes implémentées par des listes de blocs dont la somme des tailles vaut N
    #     res = 0
    #     if len(A) != len(B):
    #         raise("A and B must be of the same dimension")
    #     for k,A_k in enumerate(A):
    #         B_k = B[k]
    #         if len(A_k) != len(B_k):
    #             raise("A_k and B_k must have the same blocks")
    #         for l,A_kl in enumerate(A_k):
    #             B_kl = B_k[l]
    #             if A_kl.shape != B_kl.shape:
    #                 raise("A_k and B_k must have the same blocks")
    #             N_kl,_ = A_kl.shape
    #             for n in range(N_kl):
    #                 res = max(res,np.dot(A_kl[n,:],B_kl[n,:])/(2*N))
    #     return res

 