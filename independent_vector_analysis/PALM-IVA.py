import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from independent_vector_analysis.iva_g import iva_g
from independent_vector_analysis.helpers_iva import whiten_data
import cProfile

def make_A(K,N,seed=None):
    if seed == None:
        A = np.random.normal(0,1,(K,N,N))
    else:
        rng = np.random.default_rng(seed)
        A = rng.normal(0,1,(K,N,N))
    return A

def make_V(K,rho):
    return rho*np.ones((K,K)) + (1-rho)*np.eye(K)

def make_Sigma(K,N,V = np.zeros(1),seed=None):
    Q = np.zeros((N,K,K+10))
    mean = np.zeros(K)
    Sigma = np.zeros((N,K,K))
    if np.all(V == np.zeros(1)):
        V = np.eye(K)
    if seed==None:
        for n in range(N):
            Q[n,:,:] = np.random.multivariate_normal(mean,V,K+10).T
            Q[n,:,:] = Q[n,:,:]/np.linalg.norm(Q[n,:,:],axis=0)
            Sigma[n,:,:] = np.dot(Q[n,:,:],Q[n,:,:].T)
    else:
        for n in range(N):
            rng = np.random.default_rng(seed)
            Q[n,:,:] = rng.multivariate_normal(mean,V,K+10).T
            Q[n,:,:] = Q[n,:,:]/np.linalg.norm(Q[n,:,:],axis=0)
            Sigma[n,:,:] = np.dot(Q[n,:,:],Q[n,:,:].T)
    return Sigma

def make_S(Sigma,T):
    N,K,_ = Sigma.shape
    S = np.zeros((N,T,K))
    mean = np.zeros(K)
    for n in range(N):
        S[n,:,:] = np.random.multivariate_normal(mean,Sigma[n,:,:],T)
    return S

def sym(A):
    if A.ndim == 2:
        return (A.T+A)/2
    else:
        return (A + np.moveaxis(A,1,2))/2

def make_X(S,A):
    X = np.einsum('KMN,NTK -> MTK',A,S)
    return X

def cov_X(X):
    N,T,K = X.shape
    vec_X = np.moveaxis(X,-1,0).reshape(K*N,T)
    Lambda = vec_X.dot(vec_X.T)/T
    return Lambda

def spectral_norm(M):
    if M.ndim == 2:
        return np.linalg.norm(M,ord = 2)
    else:
        return np.max(np.linalg.norm(M,ord = 2,axis=(1,2)))

def block_diag(W):
    K,N,_ = W.shape
    W_bd = np.zeros((N,K,K*N))
    for k in range(K):
        W_bd[:,k,k*N:(k+1)*N] = W[k,:,:]
    return W_bd

def lipschitz(C,Lambda):
    return spectral_norm(C)*spectral_norm(Lambda)

def cost_WC(W,C,Lambda):
    W_bd = block_diag(W)
    W_bdT = np.moveaxis(W_bd,1,2)
    res = np.trace(np.sum(np.einsum('nkK, nKN, Ni, nij-> nkj',C,W_bd,Lambda,W_bdT),axis=0))
    res -= np.sum(np.log(np.linalg.det(C)))/2
    res -= np.sum(np.log(np.abs(np.linalg.det(W))))
    return res

def cost_reg(W,C,Lambda,alpha):
    N,K,_ = C.shape
    res = cost_WC(W,C,Lambda)
    for n in range(N):
        for k in range(K):
            res += 0.5*alpha*(C[n,k,k]-1)**2
    return res


def grad_H_W(W,C,Lambda):
    N,K,_ = C.shape
    grad = np.zeros((K,N,N))
    W_bd = block_diag(W)
    grad_tmp = np.einsum('nkK, nKN, Ni-> nki',C,W_bd,Lambda)
    for k in range(K):
        grad[k,:,:] = grad_tmp[:,k,k*N:(k+1)*N]
    return grad

def grad_H_C(W,Lambda):
    W_bd = block_diag(W)
    grad = np.einsum('nkN, Nv, nvi-> nki',W_bd,Lambda,np.moveaxis(W_bd,1,2))/2
    return sym(grad)

def grad_H_C_reg(W,C,Lambda,alpha):
    W_bd = block_diag(W)
    grad = np.einsum('nkN, Nv, nvi-> nki',W_bd,Lambda,np.moveaxis(W_bd,1,2))/2
    N,K,_ = C.shape
    for k in range(K):
        grad[:,k,k] += alpha*(C[:,k,k] - 1)
    return sym(grad)

def prox_f(W,gamma):
    K,_,_ = W.shape
    U,s,Vh = np.linalg.svd(W)
    s_new = (s + np.sqrt(s**2 + 4/gamma))/2
    diag_s = np.zeros_like(W)
    for k in range(K):
        diag_s[k,:,:] = np.diag(s_new[k,:])
    W_new = np.einsum('kNv,kvw,kwM -> kNM',U,diag_s,Vh)
    return W_new

def prox_g(C,gamma):
    N,_,_ = C.shape
    U,s,_ = np.linalg.svd(C)
    s_new = (s + np.sqrt(s**2+2/gamma))/2
    diag_s = np.zeros_like(C)
    for n in range(N):
        diag_s[n,:,:] = np.diag(s_new[n,:])
    C_new = np.einsum('nNv,nvw,nwM -> nNM',U,diag_s,np.moveaxis(U,1,2))
    return sym(C_new)

def joint_isi(W,A):
    K,N,_ = W.shape
    G_bar = np.sum(np.abs(np.einsum('knN, kNv-> knv',W,A)),axis=0)
    score = (np.sum(np.sum(G_bar/np.max(G_bar,axis=0),axis=0)-1) + np.sum(np.sum(G_bar.T/np.max(G_bar.T,axis=0),axis=0)-1))
    return score/(2*N*(N-1))

def decrease(cost):
    accr = cost[:-1] - cost[1:]
    return np.all(accr >= 0)

def sum_lambda_k(mu,lambda_0,gamma):
    return np.sum(((lambda_0 - gamma*mu/2) + np.sqrt((lambda_0 - gamma*mu/2)**2 + 2*gamma)))/2

def find_mu(lambda_0,gamma,crit = 1e-10):
        K = len(lambda_0)
        a = 2*(np.min(lambda_0) - 1)/gamma
        b = 2*np.max(lambda_0)/gamma + 1
        while b - a > crit:
            c = (a+b)/2
            value = sum_lambda_k(c,lambda_0,gamma)
            if value >= K:
                a = c 
            else:
                b = c
        return (a+b)/2   

def constrained_prox_g(C,gamma,crit = 1e-10):
    N,_,_ = C.shape
    C_new = np.zeros_like(C)
    for n in range(N):
        U,s,_ = np.linalg.svd(C[n,:,:])
        mu = find_mu(s,gamma,crit)
        s_new = ((s - gamma*mu/2) + np.sqrt((s - gamma*mu/2)**2 + 2*gamma))/2
        C_new[n,:,:] = sym(np.dot(U,np.dot(np.diag(s_new),U.T)))
    return C_new

def PALM_WC(X,gamma_c=1,max_iter=5000,criteria=10**(-6),update_scheme=(1,1),track_cost=True,track_isi=True,B=None,seed=None):
    N,_,K = X.shape
    Lambda = cov_X(X)
    C = make_Sigma(K,N,seed=seed)
    W = make_A(K,N,seed)
    N_step = 0
    if track_cost:
        cost = [cost_WC(W,C,Lambda)]
    if track_isi:
        if np.any(B == None):
            raise("you must provide B to track ISI")
        else:
            ISI = [joint_isi(W,B)]
    diff = 1
    N_updates_C, N_updates_W = update_scheme
    while diff > criteria and N_step < max_iter:
        for update in range(N_updates_C):
            grad_C = grad_H_C(W,Lambda)
            C = C - (1/gamma_c)*grad_C
            # C = prox_g(C,gamma_c)
            C = constrained_prox_g(C,gamma_c)
        for update in range(N_updates_W):
            gamma_w = 1.1*lipschitz(C,Lambda)
            grad_W = grad_H_W(W,C,Lambda)
            W_new = W - (1/gamma_w)*grad_W
            W_new = prox_f(W_new,gamma_w)
            diff = np.linalg.norm(W_new-W)
            W = W_new
        N_step += 1
        if track_cost:
            cost.append(cost_WC(W,C,Lambda))
        if track_isi:
            ISI.append(joint_isi(W,B))
    if track_cost:
        if track_isi:
            return W,C,cost,ISI
        else:
            return W,C,cost
    else:
        if track_isi:
            return W,C,ISI
        else:
            return W,C

def PALM_WC_reg(X,alpha=1,gamma_c_factor=1.1,max_iter=5000,criteria=10**(-6),update_scheme=(1,1),track_cost=True,seed=None,track_isi=True,B=None):
    N,_,K = X.shape
    Lambda = cov_X(X)
    C = make_Sigma(K,N,seed=seed)
    W = make_A(K,N,seed)
    N_step = 0
    if alpha == 0:
        gamma_c = gamma_c_factor
    else:
        gamma_c = alpha*gamma_c_factor
    if track_cost:
        cost = [cost_reg(W,C,Lambda,alpha)]
    if track_isi:
        if np.any(B == None):
            raise("you must provide B to track ISI")
        else:
            ISI = [joint_isi(W,B)]
    gamma_c = alpha*gamma_c_factor
    diff = 1
    N_updates_C, N_updates_W = update_scheme
    while diff > criteria and N_step < max_iter:
        for update in range(N_updates_C):
            grad_C = grad_H_C_reg(W,C,Lambda,alpha)
            C = C - (1/gamma_c)*grad_C
            C = prox_g(C,gamma_c)
        for update in range(N_updates_W):
            gamma_w = 1.1*lipschitz(C,Lambda)
            grad_W = grad_H_W(W,C,Lambda)
            W_new = W - (1/gamma_w)*grad_W
            W_new = prox_f(W_new,gamma_w)
            diff = np.linalg.norm(W_new-W)
            W = W_new
        N_step += 1
        if track_cost:
            cost.append(cost_reg(W,C,Lambda,alpha))
        if track_isi:
            ISI.append(joint_isi(W,B))
    if track_cost:
        if track_isi:
            return W,C,cost,ISI
        else:
            return W,C,cost
    else:
        if track_isi:
            return W,C,ISI
        else:
            return W,C

# -----------------------------------------------------------------------------------------------------

K = 10
N = 10
T = 10000
# alphas = [1]
# factors = [1.1]
alpha = 1
factor = 1.1
updates_schemes = [(1,1)] #,(1,3),(1,5),(1,10)]
# times = np.zeros((len(updates_schemes),len(alphas),len(factors)))
# scores = np.zeros((len(updates_schemes),len(alphas),len(factors)))
times = np.zeros(len(updates_schemes))
scores = np.zeros(len(updates_schemes))
N_runs = 20
for i in tqdm(range(N_runs)):
    A = make_A(K,N)
    Sigma = make_Sigma(K,N)
    S = make_S(Sigma,T)
    X = make_X(S,A)
    X_,V = whiten_data(X)
    A = np.moveaxis(A,0,-1)
    A_ = np.einsum('nNk, Nvk-> nvk', V, A)
    A_ = np.moveaxis(A_,-1,0)
    # for j,alpha in enumerate(alphas):
    #     for k,factor in enumerate(factors):
    for l,update_scheme in enumerate(updates_schemes):
        t1 = time()
        # W_f,C_f = PALM_WC_reg(X_,alpha=alpha,gamma_c_factor=factor,criteria=1e-4,update_scheme=update_scheme,track_cost=False,track_isi=False,seed=i)
        W_f,C_f = PALM_WC(X_,criteria=1e-4,update_scheme=update_scheme,track_cost=False,track_isi=False,seed=i)
        # times[j,k] += (time() - t1)/N_runs
        # scores[j,k] += joint_isi(W_f,A_)/N_runs
        times[l] += (time() - t1)/N_runs
        scores[l] += joint_isi(W_f,A_)/N_runs
print('times :',np.round_(times,2))
print('scores :',np.round_(scores,3))
# fig,ax = plt.subplots(2)
# ax[0].imshow(times,cmap='gray')
# ax[1].imshow(scores)
# plt.show()
    
# ------------------------------------------------------------------------------------------------------

# W_f,C_f,cost,ISI = PALM_WC(X_,B=A_)
# fig,ax = plt.subplots(2)
# ax[0].plot(cost)
# ax[1].plot(ISI)
# ax[1].set_yscale('log')
# print(decrease(np.array(cost)))
# plt.show()

# W_f,C_f,cost,ISI = PALM_WC_reg(X_,alpha=10,B=A_)
# fig,ax = plt.subplots(2)
# ax[0].plot(cost)
# ax[1].plot(ISI)
# ax[1].set_yscale('log')
# plt.show()

# fig,ax = plt.subplots(N,2)
# for n in range(N):
#     for k in range(K):
#         ax[n,0].plot(S[n,:100,k],c = (k/K,0,0))
#         ax[n,1].plot(X[n,:100,k],c = (0,k/K,0))
# plt.show()

def experiment(N_exp,crit):
    rho_1 = 0.2
    rho_2 = 0.8
    alpha = 1
    factor = 1.1

    rhos = [rho_1,rho_2]
    Ks = [2,5] #,10]
    Ns = [3,5,10]
    T = 10000
    results_PALM = np.zeros((len(Ks),len(Ns),N_exp))
    t_PALM = np.zeros((len(Ks),len(Ns)))
    results_IVA_G_B = np.zeros((len(Ks),len(Ns),N_exp))
    t_IVA_G_B = np.zeros((len(Ks),len(Ns)))
    results_IVA_G_grad = np.zeros((len(Ks),len(Ns),N_exp))
    t_IVA_G_grad = np.zeros((len(Ks),len(Ns)))
    for rho in rhos:
        for i,K in enumerate(Ks):
            for j,N in enumerate(Ns):
                for exp in tqdm(range(N_exp)):
                    A = make_A(K,N)
                    V = make_V(K,rho)
                    Sigma = make_Sigma(K,N,V)
                    S = make_S(Sigma,T)
                    X = make_X(S,A)
                    X_,U = whiten_data(X)
                    A = np.moveaxis(A,0,-1)
                    A_ = np.einsum('nNk, Nvk-> nvk', U, A)
                    A_ = np.moveaxis(A_,-1,0)
                    t_PALM[i,j] -= time()
                    W_PALM,_ = PALM_WC_reg(X_,alpha=200,gamma_c_factor=factor,criteria=crit,track_cost=False,track_isi=False)
                    # W_PALM,_ = PALM_WC(X_,gamma_c = 1,criteria=1e-4,track_cost=False,track_isi=False)
                    t_PALM[i,j] += time()
                    results_PALM[i,j,exp] = joint_isi(W_PALM,A_)
                    t_IVA_G_B[i,j] -= time()
                    W_B,_,_,_ = iva_g(X_,W_diff_stop=crit,max_iter=5000)
                    t_IVA_G_B[i,j] += time()
                    W_B = np.moveaxis(W_B,-1,0)
                    results_IVA_G_B[i,j,exp] = joint_isi(W_B,A_)
                    t_IVA_G_grad[i,j] -= time()
                    W_grad,_,_,_ = iva_g(X_,opt_approach='gradient',W_diff_stop=crit,max_iter=5000)
                    t_IVA_G_grad[i,j] += time()
                    W_grad = np.moveaxis(W_grad,-1,0)
                    results_IVA_G_grad[i,j,exp] = joint_isi(W_grad,A_)
        fig, ax = plt.subplots(len(Ks),len(Ns))
        fig.suptitle('ISI scores for IVA-G-B (red) and PALM (blue), rho = {}'.format(rho))
        for i in range(len(Ks)):
            for j in range(len(Ns)):
                ax[i,j].hist(results_PALM[i,j,:], bins=10,color='blue',histtype='step',label= 'mean : {:.2e}, std : {:.2e}, time : {:.2f}'.format(np.mean(results_PALM[i,j,:]),np.std(results_PALM[i,j,:]),t_PALM[i,j]/N_exp))
                ax[i,j].hist(results_IVA_G_B[i,j,:], bins=10,color='red',histtype='step',label= 'mean : {:.2e}, std : {:.2e}, time : {:.2f}'.format(np.mean(results_IVA_G_B[i,j,:]),np.std(results_IVA_G_B[i,j,:]),t_IVA_G_B[i,j]/N_exp))
                ax[i,j].hist(results_IVA_G_grad[i,j,:], bins=10,color='green',histtype='step',label= 'mean : {:.2e}, std : {:.2e}, time : {:.2f}'.format(np.mean(results_IVA_G_grad[i,j,:]),np.std(results_IVA_G_grad[i,j,:]),t_IVA_G_grad[i,j]/N_exp))
                ax[i,j].set_xscale('log')
                ax[i,j].legend(loc="upper right")
        plt.show()


# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     experiment(20)
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()

# experiment(100,1e-6)
# -----------------------------------------------------------------------------------------------------

# t0 = time()
# # B = np.random.normal(0,1,(5,10,10))
# for n in tqdm(range(1000)):
#     A = np.random.normal(0,1,(5,10,10))
#     r = spectral_norm(A)
#     if n == 653:
#         print(r)  
#         U,s,V = np.linalg.svd(A)
#         print(np.max(s))
    # isi = joint_isi(A,B)
# t = time() - t0
# print(t/1000)

# K = 4
# N = 100
# rho = 0.99
# V = make_V(K,rho)
# Sigma = make_Sigma(K,N,V)
# print(np.mean(Sigma,axis=0))
