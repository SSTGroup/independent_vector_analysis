import torch
import numpy as np
from random import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from .iva_g import iva_g
from .helpers_iva import whiten_data
from .titan_iva_g_algebra_toolbox import *
import cProfile




def make_A(K,N,seed=None):
    if seed == None:
        A = torch.randn(N,N,K)
    else:
        torch.manual_seed(seed)
        A = torch.randn(N, N, K)
    return A


def make_Sigma(K,N,rank,epsilon=1,rho_bounds=[0.4,0.6],lambda_=0.25,seed=None,normalize=False):
    
    rng = np.random.default_rng(seed)
    #if seed is not None :
    #    torch.manual_seed(seed)
    
    J = torch.ones(K, K)
    I = torch.eye(K)
    Q = torch.zeros(K, rank, N)
    mean = torch.zeros(K)
    Sigma = torch.zeros(K, K, N)
    if N == 1:
        rho = [torch.mean(rho_bounds)]
    else:
        rho = [(n/(N-1))*rho_bounds[1] + (1-(n/(N-1)))*rho_bounds[0] for n in range(N)]
    for n in range(N):
        eta = 1 - lambda_ - rho[n]
        if eta < 0 or lambda_ < 0 or rho[n] < 0:
            raise("all three coefficients must belong to [0,1]") 
        Q[:,:,n] = torch.tensor(rng.multivariate_normal(mean,I,rank).T)
        #Q[:,:,n] = torch.distributions.multivariate_normal.MultivariateNormal(mean,I).sample((rank,rank)).T
        if normalize:
            Q[:, :, n] = (Q[:, :, n].t() / torch.norm(Q[:, :, n], dim=1)).t()
            Sigma[:,:,n] = rho[n]*J + eta*I + lambda_*torch.matmul(Q[:, :, n], Q[:, :, n].t())
        else:
            Sigma[:,:,n] = rho[n]*J + eta*I + (lambda_/rank)*torch.matmul(Q[:, :, n], Q[:, :, n].t())
    for n in range(1,N):
        Sigma[:,:,n] = (1-epsilon)*Sigma[:,:,0] + epsilon*Sigma[:,:,n]
    return Sigma


def make_S(Sigma, T):
    _, K, N = Sigma.size()
    S = torch.zeros(N, T, K)
    mean = torch.zeros(K)
    for n in range(N):
        S[n,:,:] = torch.tensor(np.random.multivariate_normal(mean,Sigma[:,:,n],T))
        #S[n, :, :] = torch.normal(mean, torch.sqrt(Sigma[:, :, n]), (T, K))
    return S

def make_X(S,A):
    X = torch.einsum('NNK,NTK -> NTK',A,S)
    return X

def generate_whitened_problem(T,K,N,epsilon=1,rho_bounds=[0.4,0.6],lambda_=0.25): #, idx_W=None):
    A = make_A(K,N)
    # A = full_to_blocks(A,idx_W,K)
    Sigma = make_Sigma(K,N,rank=K+10,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False)
    S = make_S(Sigma,T)
    X = make_X(S,A)
    X_,U = whiten_data(X)
    A_ = torch.einsum('nNk,Nvk->nvk', U, A)
    return X_,A_

def identifiability_level(Sigma):
    K, _, N = Sigma.size()
    res = float('inf')
    for n in range(N):
        for m in range(N):
            if n != m:
                A = Sigma[:, :, n]
                B = Sigma[:, :, m]
                C = A @ torch.inverse(B) - torch.inverse(B @ torch.inverse(A))
                res = min(res, torch.det(C).item())
    return res

def create_clusters_W(K, N):
    if N <= 3:
        raise ValueError('N must be greater than 3 to use a clustered model')
    J = torch.ceil(torch.sqrt(torch.tensor(N)))
    all_idx = torch.arange(N)
    Idx_W = []
    for k in range(K):
        torch.random.shuffle(all_idx)
        Idx_Wk = []
        begin = 0
        while begin <= N-(2*J+1):
            cluster_size = torch.randint(2, 2*J, (1,)).item()
            Idx_Wk.append(all_idx[begin:begin+cluster_size])
            begin += cluster_size
        Idx_Wk.append(all_idx[begin:])
        Idx_W.append(Idx_Wk)
    return Idx_W






# K = 10
# N = 10
# mu=[0.6,0.7]
# lambda_=0.25
# Sigma = make_Sigma_3(K,N,rank=K+10,epsilon = 0.5,mu=mu,lambda_=lambda_)
# print('identifiability = ',identifiability_level(Sigma))

# epsilon = 0.01
# Sigma0 = make_Sigma_2(K,N,rank=K+10,mu=[0.4,0.6],lambda_=0.25)
# ident = np.zeros(99)
# for i in range(99):
#     Sigma = np.zeros((K,K,N))
#     Sigma[:,:,0] = Sigma0[:,:,0]
#     for n in range(1,N):
#         Sigma[:,:,n] = (1-(i+1)*epsilon)*Sigma0[:,:,0] + (i+1)*epsilon*Sigma0[:,:,n]
#     # Sigma = make_Sigma_3(K,N,rank=K+10,epsilon = i*0.1)
#     # print('i =',i,'identifiability = ',identifiability_level(Sigma))
#     ident[i] = identifiability_level(Sigma)
# fig,ax = plt.subplots()
# plt.suptitle('How identifiability relates to epsilon')
# plt.plot(np.linspace(epsilon,99*epsilon,99),ident)
# plt.xlabel('epsilon')
# plt.ylabel('identifiability')
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.show()






