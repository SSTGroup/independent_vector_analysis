import torch
import numpy as np
from tools import *
from torch.utils.data import Dataset
from helpers_iva import whiten_data


## Problem simumation functions 


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
        #print(rho_bounds)
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


def generate_whitened_problem(T,K,N,rho_bounds,lambda_,epsilon=1): #, idx_W=None):
    A = make_A(K,N)
    # A = full_to_blocks(A,idx_W,K)
    Sigma = make_Sigma(K,N,rank=K+10,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False)
    S = make_S(Sigma,T)
    X = make_X(S,A)
    X_,U = whiten_data(X)
    A_ = torch.einsum('nNk,Nvk->nvk', U, A)
    X_ = X_.cuda()
    A_ = A_.cuda()
    return X_,A_


def get_metaparameters(rhos,lambdas):
    metaparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            metaparameters_multiparam.append((rho_bounds,lambda_))
    return metaparameters_multiparam


class MonDataset(Dataset):
    def __init__(self, T, K, N, metaparameters_multiparam,size):
        self.T = T
        self.K = K
        self.N = N
        self.metaparameters_multiparam = metaparameters_multiparam
        self.size = size
        self.half_size = size // 2

    def __len__(self):
        # retourne la taille du dataset
        return self.size  # remplacez par la taille réelle de votre dataset

    def __getitem__(self, idx):
        # Generates a new sample from the dataset
        if idx < self.half_size:
            rho_bounds, lambda_ = self.metaparameters_multiparam[1]  # Use case 2
        else:
            rho_bounds, lambda_ = self.metaparameters_multiparam[3]  # Use case 4

        X, A = generate_whitened_problem(self.T, self.K, self.N, rho_bounds, lambda_)
        return X, A
