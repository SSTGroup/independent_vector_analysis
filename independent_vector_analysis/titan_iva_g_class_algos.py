import os
from datetime import datetime
import numpy as np
import pandas as pd
import reportlab as rl
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sc
from tqdm import tqdm
from time import time
from iva_g import iva_g
from helpers_iva import whiten_data
import cProfile
from titan_iva_g_algebra_toolbox import *
from titan_iva_g_problem_simulation import *
from titan_iva_g_reg import *


class IvaGAlgorithms:

    def __init__(self,name,legend,color,max_iter=20000,crit_ext=1e-8):
        self.name = name
        self.legend = legend
        self.color = color
        self.max_iter = max_iter
        self.crit_ext = crit_ext
        self.results = None
        self.times = None

    def set_up_for_benchmark_experiment(self,parameters_dimensions):
        self.results = np.zeros(parameters_dimensions)
        self.times = np.zeros(parameters_dimensions)

    def solve_with_isi(self,X,A):
        pass
    
    def solve_with_cost(self,X):
        pass

    def solve(self,X,Winit,Cinit):
        pass

    def fill_experiment(self,X,A,coordinates,Winit=None,Cinit=None):
        self.times[coordinates] -= time()
        W = self.solve(X,Winit,Cinit)
        self.times[coordinates] += time()
        self.results[coordinates] = joint_isi(W,A)

    def fill_from_folder(self,foldername,meta_parameters,meta_parameters_titles,common_parameters,N_exp):
        Ks,Ns = common_parameters
        dimensions = len(meta_parameters),len(Ks),len(Ns),N_exp
        self.set_up_for_benchmark_experiment(dimensions)
        for a,metaparam in enumerate(meta_parameters):
            metaparam_title = meta_parameters_titles[a]   
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    filepath = foldername + '/{}/N = {} K = {}'.format(metaparam_title,N,K)
                    if os.path.exists(filepath):
                        self.results[a,ik,jn,:] = np.fromfile(filepath+'/results_'+self.name,sep=',')
                        self.times[a,ik,jn,:] = np.fromfile(filepath+'/times_'+self.name,sep=',')

class IvaGN(IvaGAlgorithms):

    def __init__(self,color,name='IVA-G-N',legend='IVA-G-N',max_iter=20000,crit_ext=1e-6):
        super().__init__(name=name,legend=legend,color=color,max_iter=max_iter,crit_ext=crit_ext)
        self.alternated = False

    def solve(self,X,Winit,Cinit):
        _,_,K = X.shape
        for k in range(K):
            Winit[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(Winit[:, :, k] @ Winit[:, :, k].T), Winit[:, :, k])
        W,_,_,_ = iva_g(X,opt_approach='newton',W_diff_stop=self.crit_ext,
                        max_iter=self.max_iter,W_init=Winit)
        return W 

    def solve_with_isi(self,X,A):
        _,_,_,isi = iva_g(X,opt_approach='newton',jdiag_initW=False,W_diff_stop=self.crit_ext,
                        max_iter=self.max_iter,A=A)
        return isi
    
    def solve_with_cost(self,X):
        _,cost,_,_ = iva_g(X,opt_approach='newton',jdiag_initW=False,W_diff_stop=self.crit_ext,
                        max_iter=self.max_iter)
        return cost


class IvaGV(IvaGAlgorithms):

    def __init__(self,color,name='IVA-G-V',legend='IVA-G-V',max_iter=20000,crit_ext=1e-6):
        super().__init__(name=name,legend=legend,color=color,max_iter=max_iter,crit_ext=crit_ext)
        self.alternated = False

    def solve(self,X,Winit,Cinit):
        _,_,K = X.shape
        for k in range(K):
            Winit[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(Winit[:, :, k] @ Winit[:, :, k].T), Winit[:, :, k])
        W,_,_,_ = iva_g(X,opt_approach='gradient',W_diff_stop=self.crit_ext,
                        max_iter=self.max_iter,W_init=Winit)
        return W  

    def solve_with_isi(self,X,A):
        _,_,_,isi  = iva_g(X,opt_approach='gradient',jdiag_initW=False,W_diff_stop=self.crit_ext,
                        max_iter=self.max_iter,A=A)
        return isi
    
    def solve_with_cost(self,X):
        _,cost,_,_ = iva_g(X,opt_approach='gradient',jdiag_initW=False,W_diff_stop=self.crit_ext,
                        max_iter=self.max_iter)
        return cost
    
    

class PalmIvaG(IvaGAlgorithms):

    def __init__(self,color,name='palm',alpha=1,max_iter=20000,max_iter_int=50,crit_ext=1e-9,crit_int=1e-9,
                 gamma_w=0.99,gamma_c=1.99,seed=None):
        super().__init__(name=name,color=color,max_iter=max_iter,crit_ext=crit_ext)
        self.crit_int = crit_int
        self.max_iter_int = max_iter_int
        self.alpha = alpha
        self.alternated = True
        self.gamma_w = gamma_w
        self.gamma_c = gamma_c
        self.seed = seed

    def solve(self,X,Winit,Cinit):
        W,_,_ = palm_iva_g_reg(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                               crit_ext=self.crit_ext,crit_int=self.crit_int,
                               max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                               Winit=Winit,Cinit=Cinit,seed=self.seed)
        return W  
    
    def solve_with_isi(self,X,A,Winit,Cinit):
        _,_,_,isi = palm_iva_g_reg(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                               crit_ext=self.crit_ext,crit_int=self.crit_int,
                               max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                               Winit=Winit,Cinit=Cinit,seed=self.seed,track_isi=True,B=A)
    
        return isi
    
class TitanIvaG(IvaGAlgorithms):    

    def __init__(self,color,name='titan',legend='TITAN-IVA-G',alpha=1,C0=0.999,nu=0.5,max_iter=20000,max_iter_int=15,max_iter_int_C=1,
                 crit_ext=1e-9,crit_int=1e-9,gamma_w=0.99,gamma_c=1,seed=None):
        super().__init__(name=name,legend=legend,color=color,max_iter=max_iter,crit_ext=crit_ext)
        self.crit_int = crit_int
        self.max_iter_int = max_iter_int
        self.max_iter_int_C = max_iter_int_C
        self.C0 = C0
        self.nu = nu
        self.alpha = alpha
        self.alternated = True
        self.gamma_w = gamma_w
        self.gamma_c = gamma_c
        self.seed = seed

    def solve(self,X,Winit=None,Cinit=None):
        W,_,_,_ = titan_iva_g_reg(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                                     crit_ext=self.crit_ext,crit_int=self.crit_int,
                                     max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                                     max_iter_int_C=self.max_iter_int_C,C0=self.C0,nu=self.nu,
                                     Winit=Winit,Cinit=Cinit,seed=self.seed)
        return W 

    def solve_with_isi(self,X,A,Winit=None,Cinit=None):
        _,_,_,_,isi = titan_iva_g_reg(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                                     crit_ext=self.crit_ext,crit_int=self.crit_int,C0=self.C0,nu=self.nu,
                                     max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                                     Winit=Winit,Cinit=Cinit,seed=self.seed,track_isi=True,B=A)
        return isi

    def solve_with_cost(self,X,Winit=None,Cinit=None):
        _,_,_,times,cost = titan_iva_g_reg(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                                     crit_ext=self.crit_ext,crit_int=self.crit_int,C0=self.C0,nu=self.nu,
                                     max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                                     Winit=Winit,Cinit=Cinit,seed=self.seed,track_cost=True)
        return times,cost   
    
