import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from iva_g import iva_g
from helpers_iva import whiten_data
import cProfile
from titan_iva_g_algebra_toolbox import *
from titan_iva_g_problem_simulation import *



# def cost_decrease_bloc_W(W0,W1,C,Lambda):
#     W_bd0 = block_diag(W0)
#     W_bd0T = np.moveaxis(W_bd0,0,1)
#     W_bd1 = block_diag(W1)
#     W_bd1T = np.moveaxis(W_bd1,0,1)
#     delta_WLW = np.einsum('KNn, Ni, ijn -> kjn',C,W_bd0,Lambda,W_bd0T)
#     res = np.trace(np.sum(np.einsum('kKn, KNn, Ni, ijn -> kjn',C,W_bd0,Lambda,W_bd0T),axis=2))/2
#     res -= np.sum(np.log(np.linalg.det(np.moveaxis(C,2,0))))/2
#     res -= np.sum(np.log(np.abs(np.linalg.det(np.moveaxis(W,2,0)))))
#     return res



# def sum_lambda_k(mu,lambda_0,gamma):
#     return np.sum(((lambda_0 - gamma*mu/2) + np.sqrt((lambda_0 - gamma*mu/2)**2 + 2*gamma)))/2

# def find_mu(lambda_0,gamma,crit = 1e-10):
#         K = len(lambda_0)
#         a = 2*(np.min(lambda_0) - 1)/gamma
#         b = 2*np.max(lambda_0)/gamma + 1
#         while b - a > crit:
#             c = (a+b)/2
#             value = sum_lambda_k(c,lambda_0,gamma)
#             if value >= K:
#                 a = c 
#             else:
#                 b = c
#         return (a+b)/2   

# def constrained_prox_g(C,gamma,crit = 1e-10):
#     _,_,N = C.shape
#     C_new = np.zeros_like(C)
#     for n in range(N):
#         U,s,_ = np.linalg.svd(C[:,:,n])
#         mu = find_mu(s,gamma,crit)
#         s_new = ((s - gamma*mu/2) + np.sqrt((s - gamma*mu/2)**2 + 2*gamma))/2
#         C_new[:,:,n] = sym(np.dot(U,np.dot(np.diag(s_new),U.T)))
#     return C_new

# def palm_iva_g(X,c_c=1,gamma_w=0.9,max_iter=5000,W_diff_stop=10**(-6),update_scheme=(1,1),track_cost=True,track_isi=True,track_diff=False,B=None,seed=None):
#     N,_,K = X.shape
#     Lambda = cov_X(X)
#     lam = spectral_norm(Lambda)
#     C = make_Sigma(K,N,seed=seed)
#     W = make_A(K,N,seed)
#     N_step = 0
#     if track_cost:
#         cost = [cost_iva_g(W,C,Lambda)]
#     if track_isi:
#         if np.any(B == None):
#             raise("you must provide B to track ISI")
#         else:
#             ISI = [joint_isi(W,B)]
#     if track_diff:
#         diffs = []
#     diff = np.infty
#     N_updates_C, N_updates_W = update_scheme
#     while diff > W_diff_stop and N_step < max_iter:
#         W_old = W
#         C_old = C
#         for update in range(N_updates_W):
#             c_w = gamma_w/lipschitz(C,lam)
#             grad_W = grad_H_W(W,C,Lambda)
#             W = W - c_w*grad_W
#             W = prox_f(W,c_w)
#         for update in range(N_updates_C):
#             grad_C = grad_H_C(W,Lambda)
#             C = C - c_c*grad_C
#             C = prox_g(C,c_c)
#         diff = max(diff_criteria(W,W_old),diff_criteria(C,C_old))
#         if track_diff:
#             diffs.append(diff)
#             # C = constrained_prox_g(C,gamma_c)
#         N_step += 1
#         if track_cost:
#             cost.append(cost_iva_g(W,C,Lambda))
#         if track_isi:
#             ISI.append(joint_isi(W,B))
#     if N_step < max_iter:
#         met_limit = True
#     else:
#         met_limit = False
#     if track_diff:
#         if track_cost:
#             if track_isi:
#                 return W,C,met_limit,cost,ISI,diffs
#             else:
#                 return W,C,met_limit,cost,diffs
#         else:
#             if track_isi:
#                 return W,C,met_limit,ISI,diffs
#             else:
#                 return W,C,met_limit,diffs
#     else:
#         if track_cost:
#             if track_isi:
#                 return W,C,met_limit,cost,ISI
#             else:
#                 return W,C,met_limit,cost
#         else:
#             if track_isi:
#                 return W,C,met_limit,ISI
#             else:
#                 return W,C,met_limit

# N = 5
# K = 1

# W = make_A(K,N)
# W[0:2,2:5,0]=0
# W[2:5,0:2,0]=0

# W_1 = prox_f(W,2)
# tmp = W[:,:,0]
# tmp1 = tmp[0:2,0:2]
# tmp2 = tmp[2:5,2:5]
# W_11 = prox_f(np.expand_dims(tmp1,axis=0),2)
# W_12 = prox_f(np.expand_dims(tmp2,axis=0),2)

# print(W_1,W_12,W_11)
# print(W,tmp,tmp2)

# import matplotlib.pyplot as plt
# import numpy as np

# # Création du repère orthonormé
# fig, ax = plt.subplots()
# ax.axhline(0, color='black',linewidth=0.5)
# ax.axvline(0, color='black',linewidth=0.5)
# ax.set_aspect('equal', adjustable='box')

# # Configuration des graduations
# plt.xticks(np.arange(0, 1.2, 0.2))
# plt.yticks(np.arange(0, 1.2, 0.2))

# # Tracé du carré
# square = plt.Rectangle((0, 0), 1, 1, fill=True, color=('gray'), alpha=0.0)
# ax.add_patch(square)

# # Tracé de la droite y + x = 1
# x_vals1 = np.linspace(0, 1, 100)
# y_vals1 = 1.25 - 2*x_vals1
# plt.plot(x_vals1, y_vals1, label='corruption')

# # Tracé de la droite y + 2x = 1.5
# x_vals2 = np.linspace(0, 1, 100)
# y_vals2 = 1 - x_vals2
# plt.plot(x_vals2, y_vals2, label='tyranny')

# x_vals3 = np.linspace(0,1,100)
# y_vals3 = 1.5 - 2*x_vals3
# plt.plot(x_vals1, y_vals3, label='xenophobia')

# plt.fill_between(x_vals3, y_vals3, 1, color='g', alpha=0.2)  # Au-dessus de la ligne bleue
# plt.fill_between(x_vals2, y_vals2, 1, color='y', alpha=0.4)  # AU dessus de la ligne jaune
# plt.fill_between(x_vals1, y_vals1, 0, color='b', alpha=0.2)  # En dessous de la ligne verte


# # Affichage de la légende
# plt.legend()

# # Affichage du graphe
# plt.xlabel('$p_{RF}$')
# plt.ylabel('$p_{CC}$')
# plt.title('feasibility set')
# plt.grid(False)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.show()



# algo = TitanIvaG(color=(0,0,0),crit_ext=1e-9,crit_int=1e-9,max_iter_int=15)
# K = 5
# N = 5
# X,A = generate_whitened_problem(10000,K,N,rho_bounds=[0.6,0.7])
# times,cost = algo.solve_with_cost(X)
# plt.rcParams['text.usetex'] = True
# fig,ax = plt.subplots()
# ax.plot(cost-cost[-1],color=algo.color)
# ax.grid(which='both')
# # plt.suptitle('TITAN-IVA-G',fontsize=20)
# ax.set_xlabel('Iterations $i$',fontsize=30)
# ax.set_ylabel('$J_{IVA-G}^{Reg}(W^{(i)},C^{(i)}) - J^*$',fontsize=20)
# ax.set_yscale('log')
# # plt.legend(loc=1,fontsize=6)
# # plt.show()
# fig.savefig('cost iterations',dpi=200,bbox_inches='tight')
# fig2,ax2 = plt.subplots()
# ax2.plot(times,cost-cost[-1],color=algo.color)
# ax2.grid(which='both')
# # plt.suptitle('TITAN-IVA-G',fontsize=20)
# ax2.set_xlabel('Time (in s.)',fontsize=30)
# ax2.set_ylabel('$J_{IVA-G}^{Reg}(W^{(i)},C^{(i)}) - J^*$',fontsize=20)
# ax2.set_yscale('log')
# # plt.legend(loc=1,fontsize=6)
# fig2.savefig('cost time',dpi=200,bbox_inches='tight')
    

# algo = TitanIvaG(color=(0,0,0),crit_ext=1e-9,crit_int=1e-9,max_iter_int=15)
# K = 5
# N = 10
# X,A = generate_whitened_problem(10000,K,N,rho_bounds=[0.6,0.7])
# Winit = make_A(K,N)
# Cinit = make_Sigma(K,N,rank=K+10)
# t_dec1 = -time()
# _,_,_,_,isi_dec1,shifts_W_dec1 = titan_iva_g_reg(X,track_isi=True,crit_int=1e-10,crit_ext=1e-10,
#                                          B=A,max_iter_int=15,gamma_w=4,
#                                          adaptative_gamma_w=True,Cinit=Cinit,Winit=Winit)
# t_dec1 += time()
# t_dec2 = -time()
# _,_,_,_,isi_dec2,shifts_W_dec2 = titan_iva_g_reg(X,track_isi=True,crit_int=1e-10,crit_ext=1e-10,
#                                          B=A,max_iter_int=15,gamma_w=4,gamma_w_decay=0.8,
#                                          adaptative_gamma_w=True,Cinit=Cinit,Winit=Winit)
# t_dec2 += time()
# t = -time()
# _,_,_,_,isi,shifts_W = titan_iva_g_reg(X,track_isi=True,crit_int=1e-10,crit_ext=1e-10,
#                                          B=A,max_iter_int=15,Cinit=Cinit,Winit=Winit)
# t += time()
# fig,ax = plt.subplots()
# ax.plot(shifts_W_dec1,color='r',label='time BT 1 : {}, res BT 1: {}'.format(t_dec1,isi_dec1[-1]))
# ax.plot(shifts_W_dec2,color='g',label='time BT 2: {}, res BT 2: {}'.format(t_dec2,isi_dec2[-1]))
# ax.plot(shifts_W,color='b',label='time normal : {}, res normal : {}'.format(t,isi[-1]))
# ax.grid(which='both')
# ax.set_xlabel('Iterations $i$',fontsize=30)
# ax.set_ylabel('shifts of W',fontsize=20)
# ax.legend(loc=1)
# plt.show()
