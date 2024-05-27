import os
from datetime import datetime
import numpy as np
import scipy as sc
import pandas as pd
import reportlab as rl
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from time import time
from iva_g import iva_g
from helpers_iva import whiten_data
import cProfile
from titan_iva_g_algebra_toolbox import *
from titan_iva_g_problem_simulation import *
from titan_iva_g_reg import *
from titan_iva_g_class_exp import *
from titan_iva_g_class_algos import *
#------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]
# identifiability_levels = [1e-2,1e-1,1])
# identifiability_levels_names = ['low identifiability','medium identifiability','high identifiability']
Ks = [5,10]
Ns = [3,5,10]

common_parameters = [Ks,Ns]
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D']

def get_metaparameters(rhos,lambdas):
    metaparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            metaparameters_multiparam.append((rho_bounds,lambda_))
    return metaparameters_multiparam
         
# ------------------------------------------------------------------------------------------------------------------------------                
# ------------------------------------------------------------------------------------------------------------------------------------------

label_size = 60
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
plt.rcParams['text.usetex'] = True

metaparameters_multiparam = get_metaparameters(rhos,lambdas)

algo_titan = TitanIvaG((0,0.5,0),name='titan',gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10)
algo_iva_g_n = IvaGN((0,0,1),name='iva_g_n',crit_ext=1e-7)
algo_iva_g_v = IvaGV((1,0,0),name='iva_g_v',crit_ext=1e-6)


algos = [algo_titan,algo_iva_g_v,algo_iva_g_n]


exp2 = ComparisonExperimentIvaG('multiparameter benchmark',algos,metaparameters_multiparam,metaparameters_titles_multiparam,
                                common_parameters,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=1,charts=False,legend=False)
exp2.compute()
# exp2.get_data_from_folder('2024-05-16_02-22')
# exp2.make_table()
# exp2.make_charts(full=True)


# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     exp2.compute()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()