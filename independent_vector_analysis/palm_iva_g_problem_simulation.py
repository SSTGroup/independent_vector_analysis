import numpy as np
from random import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from iva_g import iva_g
from helpers_iva import whiten_data
from palm_iva_g_algebra_toolbox import *
import cProfile


def make_A(K,N,seed=None):
    if seed == None:
        A = np.random.normal(0,1,(N,N,K))
    else:
        rng = np.random.default_rng(seed)
        A = rng.normal(0,1,(N,N,K))
    return A

def make_Sigma(K,N,rank,epsilon=1,rho_bounds=[0.4,0.6],lambda_=0.25,seed=None,normalize=False):
    rng = np.random.default_rng(seed)
    J = np.ones((K,K))
    I = np.eye(K)
    Q = np.zeros((K,rank,N))
    mean = np.zeros(K)
    Sigma = np.zeros((K,K,N))
    if N == 1:
        rho = [np.mean(rho_bounds)]
    else:
        rho = [(n/(N-1))*rho_bounds[1] + (1-(n/(N-1)))*rho_bounds[0] for n in range(N)]
    for n in range(N):
        eta = 1 - lambda_ - rho[n]
        if eta < 0 or lambda_ < 0 or rho[n] < 0:
            raise("all three coefficients must belong to [0,1]")
        Q[:,:,n] = rng.multivariate_normal(mean,I,rank).T
        if normalize:
            Q[:,:,n] = (Q[:,:,n].T/np.linalg.norm(Q[:,:,n],axis=1)).T
            Sigma[:,:,n] = rho[n]*J + eta*I + lambda_*np.dot(Q[:,:,n],Q[:,:,n].T)
        else:
            Sigma[:,:,n] = rho[n]*J + eta*I + (lambda_/rank)*np.dot(Q[:,:,n],Q[:,:,n].T)
    for n in range(1,N):
        Sigma[:,:,n] = (1-epsilon)*Sigma[:,:,0] + epsilon*Sigma[:,:,n]
    return Sigma

def make_S(Sigma,T):
    _,K,N = Sigma.shape
    S = np.zeros((N,T,K))
    mean = np.zeros(K)
    for n in range(N):
        S[n,:,:] = np.random.multivariate_normal(mean,Sigma[:,:,n],T)
    return S

def make_X(S,A):
    X = np.einsum('MNK,NTK -> MTK',A,S)
    return X

def generate_whitened_problem(T,K,N,epsilon=1,rho_bounds=[0.4,0.6],lambda_=0.25): #, idx_W=None):
    A = make_A(K,N)
    # A = full_to_blocks(A,idx_W,K)
    Sigma = make_Sigma(K,N,rank=K+10,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False)
    S = make_S(Sigma,T)
    X = make_X(S,A)
    X_,U = whiten_data(X)
    A_ = np.einsum('nNk,Nvk->nvk',U,A)
    return X_,A_

def identifiability_level(Sigma):
    K,_,N = Sigma.shape
    res = np.inf
    for n in range(N):
        for m in range(N):
            if n != m:
                A = Sigma[:,:,n]
                B = Sigma[:,:,m]
                C = A*np.linalg.inv(B) - np.linalg.inv(B*np.linalg.inv(A))
                res = min(res,np.linalg.det(C))
                # res = min(res,np.min(np.linalg.eigvalsh(C)))
    return res

def create_clusters_W(K,N):
    if N <= 3:
        raise('N must be greater than 3 to use a clustered model')
    J = np.ceil(np.sqrt(N))
    all_idx = np.arange(N)
    Idx_W = []
    for k in range(K):
        np.random.shuffle(all_idx)
        Idx_Wk = []
        begin = 0
        while begin <= N-(2J+1):
            # Si begin = N - (2J+1), on arrive au plus à begin+cluster_size = N-2, donc on aura un dernier bloc de taille 2
            cluster_size = randint(2,2J)
            # J est toujours supérieur ou égal à 2 donc les tailles de cluster peuvent au moins aller jusqu'à 4
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






