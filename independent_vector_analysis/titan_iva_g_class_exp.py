import os
from datetime import datetime
import numpy as np
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
from titan_iva_g_class_algos import *
   
class ComparisonExperimentIvaG:
#On classe les résultats et les graphes dans une arborescence de 2 niveaux :
#un premier niveau de meta-paramètres qui dépendent du mode d'expérience 
#(donc un sous-dossier par combinaison de MP)
#puis un second niveau de paramètres commun (en l'occurrence K et N), c'est la que sont les graphes de comparaison
#Si on veut faire varier d'autres paramètres au niveau des algos, on définit plusieurs algorithmes séparés ! 


# L'idée de cette classe est de créer un objet "expérience" qui est déterminé par son nom 
# (lié au mode de l'expérience, mais pas que, à voir au cas par cas), par la date à laquelle
# elle est lancée, et qui contient/fabrique les résultats sous forme de données dans les algos 
# qu'elle implique ou dans des dossiers qui peuvent ou pas contenir des graphes. On veut pouvoir
# recréer un objet expérience à partir d'un dossier pour retravailler les données calculées et les présenter
# différemment par exemple
      
    def __init__(self,name,algos,meta_parameters,meta_parameters_titles,common_parameters,mode='multiparam',
                 T=10000,N_exp=100,table=False,table_fontsize=5,median=False,charts=False,legend=True,
                 legend_fontsize=5,title_fontsize=10):  
        self.algos = algos
        self.N_exp = N_exp
        self.mode = mode
        self.meta_parameters = meta_parameters
        self.meta_parameters_titles = meta_parameters_titles
        self.common_parameters = common_parameters
        # self.common_parameters_names = common_parameters_names
#parameters_name est une liste dont chaque élément est un tableau contenant les valeurs que prennent ces paramètres
        self.name = name
        self.T = T
        self.table = table
        self.table_fontsize = table_fontsize
        self.median = median
        self.charts = charts
        self.legend = legend
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize
        now = datetime.now()
        self.date = now.strftime("%Y-%m-%d_%H-%M")
         
    def get_data_from_folder(self,date):
        self.date = date
        foldername = self.date + ' ' + self.name
        Ks,Ns = self.common_parameters
        dimensions = (len(self.meta_parameters),len(Ks),len(Ns),self.N_exp)
        for algo in self.algos:
            algo.set_up_for_benchmark_experiment(dimensions)
            algo.fill_from_folder(foldername,self.meta_parameters,self.meta_parameters_titles,self.common_parameters,self.N_exp)

    def best_perf(self,criterion='results'):
        Ks,Ns = self.common_parameters
        best_perfs = np.zeros((len(self.meta_parameters),len(Ks),len(Ns)))
        for a,meta_param in enumerate(self.meta_parameters):
                for ik,K in enumerate(Ks):
                    for jn,N in enumerate(Ns):
                        if criterion == 'results':
                            perfs = [np.mean(algo.results[a,ik,jn,:]) for algo in self.algos]
                        else:
                            perfs = [np.mean(algo.times[a,ik,jn,:]) for algo in self.algos]
                        best_perfs[a,ik,jn] = min(perfs)
        return best_perfs
   
    def make_table(self,tols=(1e-4,1e-2)):
        output_folder = self.date + ' ' + self.name   
        Ks,Ns = self.common_parameters
        n_cols = len(Ks)*len(Ns)
        best_results = self.best_perf(criterion='results')
        best_times = self.best_perf(criterion = 'times')
        tol_res,tol_time = tols
        # We consider that results_algo come from the same experiment
        filename = 'table results.txt' #+ algo.name + '.txt'
        output_path = os.path.join(output_folder, filename)
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as file:
            file.write('\\begin{table}[h!]\n\\caption{'+'blablabla'+'}\n\\vspace{0.4cm}\n')
            file.write('\\fontsize{{{}pt}}{{{}pt}}\selectfont\n'.format(self.table_fontsize,self.table_fontsize))
            file.write('\\begin{{tabular}}{{{}}}\n'.format('cm{0.5cm}m{0.5cm}'+n_cols*'c'))
            file.write('& &')
            for K in Ks:
                file.write(' & \\multicolumn{{{}}}{{c}}{{$K$ = {}}}'.format(len(Ns),K))
            file.write('\\\\\n')
            for ik,K in enumerate(Ks):
                file.write(' \\cmidrule(lr){{{}-{}}}'.format(4+ik*len(Ns),3+(ik+1)*len(Ns)))
            file.write('\n')
            file.write('& &')
            for K in Ks:
                for N in Ns:
                    file.write(' & $N$ = {}'.format(N))
            file.write('\\\\\n')
            for algo_index,algo in enumerate(self.algos):
                file.write('\\midrule\n')
                file.write('\\multirow{{{}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\small{{\\textbf{{{}}}}}}}}}'.format(3*len(self.meta_parameters),algo.legend))
                for a,metaparam in enumerate(self.meta_parameters):
                    file.write('& \\multirow{{{}}}{{*}}{{\\begin{{tabular}}{{c}} {} \\end{{tabular}}}}& $\\mu_{{\\rm ISI}}$'.format(3+2*self.median,self.meta_parameters_titles[a]))
                    for ik,K in enumerate(Ks):
                        for jn,N in enumerate(Ns):
                            if np.mean(algo.results[a,ik,jn,:]) <= best_results[a,ik,jn] + tol_res:
                                file.write(' & \\textbf{{{:.2E}}}'.format(np.mean(algo.results[a,ik,jn,:])))
                            else:
                                file.write(' & {:.2E}'.format(np.mean(algo.results[a,ik,jn,:])))
                    file.write('\\\\\n')
                    if self.median:
                        file.write('& & $\\widehat{\\mu}_{\\rm ISI}$')
                        for ik,K in enumerate(Ks):
                            for jn,N in enumerate(Ns):
                                file.write(' & {:.2E}'.format(np.median(algo.results[a,ik,jn,:])))
                        file.write('\\\\\n')
                    file.write('& & $\\sigma_{\\rm ISI}$')
                    for ik,K in enumerate(Ks):
                        for jn,N in enumerate(Ns):
                            file.write(' & {:.2E}'.format(np.std(algo.results[a,ik,jn,:])))
                    file.write('\\\\\n')
                    if self.median:
                        file.write('& & $\\widehat{\\sigma}_{\\rm ISI}$')
                        for ik,K in enumerate(Ks):
                            for jn,N in enumerate(Ns):
                                file.write(' & {:.2E}'.format(np.median(np.abs(algo.results[a,ik,jn,:]-np.mean(algo.results[a,ik,jn,:])))))
                        file.write('\\\\\n')
                    file.write('& & $\\mu_T$')
                    for ik,K in enumerate(Ks):
                        for jn,N in enumerate(Ns):
                            if np.mean(algo.times[a,ik,jn,:]) <= best_times[a,ik,jn] + tol_time:
                                file.write(' & \\textit{{\\textbf{{{:.2f}}}}}'.format(np.mean(algo.times[a,ik,jn,:])))
                            else:
                                file.write(' & {:.2f}'.format(np.mean(algo.times[a,ik,jn,:])))
                    file.write('\\\\\n')
                    if a == len(self.meta_parameters)-1:
                        file.write('\\bottomrule\n')
                    else:
                        file.write('\\cmidrule(lr){{2-{}}}'.format(3+n_cols))
                    file.write('\\\\\n')
            file.write('\\end{tabular}\n\\end{table}')

    def make_charts(self):
        output_folder = self.date + ' ' + self.name
        Ks,Ns = self.common_parameters
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    os.makedirs(output_folder+'/charts/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K))
                    fig,ax = plt.subplots()
                    ax.set_xlabel('$T$ (s.)',fontsize=self.title_fontsize,labelpad=0)
                    ax.set_ylabel('$ISI$ score',fontsize=self.title_fontsize,labelpad=0)
                    for algo in self.algos:
                        ax.errorbar(np.mean(algo.times[a,ik,jn,:]),np.mean(algo.results[a,ik,jn,:]),
                                                yerr=np.std(algo.results[a,ik,jn,:]),xerr=np.std(algo.times[a,ik,jn,:]),
                                                color=algo.color,label=algo.legend,elinewidth=2.5)
                    ax.set_yscale('log')
                    ax.grid(which='both')
                    if self.legend:
                        fig.legend(loc=2,fontsize=self.legend_fontsize)
                    filename = 'comparison {} N = {} K = {}'.format(self.meta_parameters_titles[a],N,K)
                    output_path = os.path.join(output_folder+'/charts/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K), filename)
                    fig.savefig(output_path,dpi=200,bbox_inches='tight')
                    plt.close(fig)
 
    def store_in_folder(self):
        output_folder = self.date + ' ' + self.name
        os.makedirs(output_folder,exist_ok=True)
        Ks,Ns = self.common_parameters
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    os.makedirs(output_folder+'/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K))
                    for algo in self.algos: 
                        algo.results[a,ik,jn,:].tofile(output_folder+'/{}/N = {} K = {}/results_{}'.format(self.meta_parameters_titles[a],N,K,algo.name),sep=',')
                        algo.times[a,ik,jn,:].tofile(output_folder+'/{}/N = {} K = {}/times_{}'.format(self.meta_parameters_titles[a],N,K,algo.name),sep=',')
        if self.charts:
            self.make_charts()
        if self.table:
            self.make_table()
                   
    def compute(self):
        Ks,Ns = self.common_parameters
        dimensions = (len(self.meta_parameters),len(Ks),len(Ns),self.N_exp)
        for algo in self.algos:
            algo.set_up_for_benchmark_experiment(dimensions)
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    for exp in range(self.N_exp):
                        if self.mode == 'identifiability':
                            X,A = generate_whitened_problem(self.T,K,N,epsilon=metaparam)
                        elif self.mode == 'multiparam':
                            rho_bounds,lambda_ = metaparam
                            X,A = generate_whitened_problem(self.T,K,N,rho_bounds=rho_bounds,lambda_=lambda_)
                        Winit = make_A(K,N)
                        Cinit = make_Sigma(K,N,rank=K+10)
                        for algo in self.algos:
                            algo.fill_experiment(X,A,(a,ik,jn,exp),Winit.copy(),Cinit.copy())
                            print(a,ik,jn,algo.name + ' : ',algo.results[a,ik,jn,exp],algo.times[a,ik,jn,exp] )
        self.store_in_folder()

    def draw_isi_evolutions(self):
        output_folder = self.date + ' ' + self.name
        os.makedirs(output_folder,exist_ok=True)
        Ks,Ns = self.common_parameters
        for a,metaparam in enumerate(self.meta_parameters):
            fig_global,axes = plt.subplots(len(Ks),len(Ns))
            fig_global.supxlabel('Iteration (external loop)',fontsize=self.title_fontsize)
            fig_global.supylabel('ISI score',fontsize=self.title_fontsize)
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    if ik == 0:
                        axes[ik,jn].set_title('N = {}'.format(N),fontsize=self.title_fontsize)
                    if jn == 0:
                        axes[ik,jn].set_ylabel('K = {}'.format(K),fontsize=self.title_fontsize)
                    os.makedirs(output_folder+'/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K))
                    fig,ax = plt.subplots()
                    fig.supxlabel('Iteration (external loop)',fontsize=self.title_fontsize)
                    fig.supylabel('ISI score',fontsize=self.title_fontsize)
                    ax.set_yscale('log')
                    axes[ik,jn].set_yscale('log')
                    X,A = generate_whitened_problem(self.T,metaparam,K,N,mode=self.mode)
                    Winit = make_A(K,N)
                    Cinit = make_Sigma(K,N)
                    for algo in self.algos:
                        t = -time()
                        isi = algo.solve_with_isi(self,X,A,Winit,Cinit)
                        t += time()
                        res = isi[-1]
                        ax.plot(isi,color=algo.color,label=algo.legend +' time = {:.2E}, ISI = {:.3f}'.format(t,res),linewidth=0.5)
                        axes[ik,jn].plot(isi,color=algo.color,label=algo.legend +' time = {:.2E}, ISI = {:.3f}'.format(t,res),linewidth=0.5)
                    ax.legend(loc=1,fontsize=self.legend_fontsize)
                    for extension in ['.eps','.png']:
                        filename = 'isi evolutions' + extension
                        output_path = os.path.join(output_folder+'/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K), filename)
                        fig.savefig(output_path,dpi=200)
            fig_global.subplots_adjust(wspace=0.2,hspace=0.4)
            fig_global.legend(loc=1,fontsize=self.legend_fontsize)
            fig_global.savefig(output_path,dpi=200)     



    






                    

        

