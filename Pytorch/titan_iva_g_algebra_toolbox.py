import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from .iva_g import iva_g
from .helpers_iva import whiten_data
from concurrent.futures import ThreadPoolExecutor
import cProfile





def sym(A):
    if A.ndim == 2:
        return (A.transpose(0, 1) + A) / 2
    else:
        return (A + torch.moveaxis(A, 0, 1)) / 2
    
""" def cov_X(X):
    N, T, K = X.size()
    vec_X = X.permute(2, 0, 1).reshape(K * N, T)
    Lambda = torch.matmul(vec_X, vec_X.t()) / T
    return Lambda """

def cov_X(X):
    _, T, _ = X.size()
    Rx = torch.einsum('NTK,MTJ->KJNM', X, X) / T
    return Rx


def spectral_norm(M):
    if M.dim() == 2:
        return torch.linalg.norm(M, ord=2)
    else:
        return torch.max(torch.linalg.norm(M, ord=2,dim=(0,1)))
    

""" def spectral_norm_extracted(Lambda, K, N):
    norms = []
    device = Lambda.device  # Get the device of the input tensor
    for j in range(K):
        norms.append(torch.linalg.norm(Lambda[:, j * N:(j + 1) * N].to(device), ord=2))  # Move slice of Lambda to GPU
    return torch.max(torch.tensor(norms).to(device))  # Move the list of norms to GPU and compute the maximum
 """
""" def spectral_norm_extracted(Rx, K, N):
    reshaped_Rx = Rx.view(K, K*N, N)  # Remodelage du tenseur Rx
    norms = torch.norm(reshaped_Rx, p=2, dim=(1, 2))  # Calcul des normes L2
    return torch.max(norms)  # Renvoyer la norme spectrale maximale """

def spectral_norm_extracted(Rx,K,N):
    return torch.max(torch.norm(torch.reshape(Rx,(K,K*N,N)),p=2,dim=(1,2)))


def smallest_singular_value(C):
    _, s, _ = torch.svd(C.permute(2, 0, 1))
    return torch.min(s)

def block_diag(W):
    N, N, K = W.size()
    W_bd = torch.zeros(K, K*N, N,device='cuda')
    for k in range(K):
        W_bd[k, k*N:(k+1)*N, :] = W[:, :, k].t()
    return W_bd

def blocks_to_full(W_blocks, K, N):
    W_full = torch.zeros(N, N, K,device='cuda')
    for k, W_k in enumerate(W_blocks):
        begin = 0
        for W_kl in W_k:
            n_l = W_kl.size(0)
            W_full[begin:begin+n_l, begin:begin+n_l, k] = W_kl
            begin += n_l
    return W_full

def full_to_blocks(W_full, idx_W, K):
    W_blocks = []
    for k in range(K):
        W_k = []
        L_k = len(idx_W[k])
        for l in range(L_k):
            W_kl = W_full[idx_W[k][l], idx_W[k][l], k]
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

def joint_isi(W, A):
    N, _, _ = W.size()
    G_bar = torch.sum(torch.abs(torch.einsum('nNk,Nvk->nvk', W, A)), dim=2)
    score = (torch.sum(torch.sum(G_bar / torch.max(G_bar, dim=0)[0], dim=0) - 1) +
             torch.sum(torch.sum(G_bar.t() / torch.max(G_bar.t(), dim=0)[0], dim=0) - 1))
    return score / (2 * N * (N - 1))

def decrease(cost, verbose=0):
    accr = torch.tensor(cost[:-1]) - torch.tensor(cost[1:])
    if torch.all(accr >= 0):
        return True
    else:
        if verbose >= 1:
            for i in range(len(accr)):
                if accr[i] < 0:
                    print("increase at index :", i)
                    print("an increase of :", -accr[i])
                    break
        return False
    
def diff_criteria(A, B, mode='full'):
    if mode == 'full':
        if A.shape != B.shape:
            raise ValueError("A and B must be of the same dimension")
        elif A.ndim < 2 or A.ndim > 3:
            raise ValueError("Only tensors of order 2 or 3 are accepted")
        res = 0
        D = A - B

        max_norm = torch.max(torch.sum(D ** 2, dim=1))
        return max_norm / (2 * A.size(0))


    elif mode == 'blocks':
        res = 0
        if len(A) != len(B):
            raise ValueError("A and B must be of the same dimension")
        for k, A_k in enumerate(A):
            B_k = B[k]
            if len(A_k) != len(B_k):
                raise ValueError("A_k and B_k must have the same blocks")
            for l, A_kl in enumerate(A_k):
                B_kl = B_k[l]
                if A_kl.shape != B_kl.shape:
                    raise ValueError("A_k and B_k must have the same blocks")
                N_kl, _ = A_kl.shape
                for n in range(N_kl):
                    res = max(res, torch.dot(A_kl[n, :], B_kl[n, :]) / (2 * N_kl))
        return res

        

  
    
# # fig,ax = plt.subplots((3,2))

# # ax[0,0].plot()
    
# # Définition des fonctions affines
# def f1(x):
#     return min(8 + x,20)

# def f2(x):
#     return min(6 + 2*x/3,20)

# def f3(x):
#     return min(2 + 3*x/4,20)

# def f4(x):
#     return min(10 +x/2,20)

# def f5(x):
#     return min(12+x,20)

# def f6(x):
#     return min(7+2*x,20)

# # Création de la grille de sous-graphiques
# fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)
# names = ['statistics','algebra','measure theory','optimisation','machine learning','complex analysis']
# # Boucle pour tracer chaque fonction sur son sous-graphique
# for i, func in enumerate([f1, f2, f3, f4, f5, f6]):
#     # Définition du domaine
#     x = np.linspace(0,20,1000)
    
#     # Tracer la fonction sur le sous-graphique correspondant
#     ax = axs[i // 3, i % 3]
#     ax.plot(x, [func(xi) for xi in x], label= names[i])
#     ax.grid(True)
#     ax.axhline(0, color='black',linewidth=2)
#     ax.axhline(20, color='black',linewidth=1)
#     ax.axhline(10, color='black',linewidth=1)
#     ax.axvline(0, color='black',linewidth=2)
#     ax.set_title(names[i])

# # Affichage des labels et du titre
# fig.suptitle('Work rentability diagrams')
# plt.tight_layout()
# plt.show()