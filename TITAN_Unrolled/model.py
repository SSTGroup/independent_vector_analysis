import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functions import *
from tools import *



class AlphaNetwork(nn.Module):
    def __init__(self, input_dim):
        super(AlphaNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1,device='cuda:0')
        #print(input_dim)
        self.activation = nn.Softmax()

    def forward(self, W, C):
        x = torch.cat((W.flatten(), C.flatten()), dim=0)
        #print('W', W.device)
        #print('C', C.device)
        #print("x", x.device)
        alpha = self.activation(self.fc1(x))
        return alpha
    


class DeterministicBlock(nn.Module):
    def __init__(self, Rx, rho_Rx, gamma_c, gamma_w, eps, nu, zeta):
        super(DeterministicBlock, self).__init__()
        self.Rx = Rx
        self.rho_Rx = rho_Rx
        self.gamma_c = gamma_c
        self.gamma_w = gamma_w
        self.eps = eps 
        self.nu = nu
        self.zeta = zeta
        self.device = 'cuda:0'    

    def forward(self, W, C, alpha, L_w_prev):
        K = W.shape[2]
        alpha = alpha.to(self.device)
        #print(alpha.device)

        l_sup = max((self.gamma_w*alpha)/(1-self.gamma_w),self.rho_Rx*2*K*(1+torch.sqrt(2/(alpha*self.gamma_c))))
        C0 = min(self.gamma_c**2/K**2,(alpha*self.gamma_w/((1+self.zeta)*(1 - self.gamma_w)*l_sup)),(self.rho_Rx/((1+self.zeta)*l_sup)))
        l_inf = (1+self.zeta)*C0*l_sup

        c_c = self.gamma_c / alpha
        beta_c = torch.sqrt(C0*self.nu*(1-self.nu))
        L_w = max(l_inf,lipschitz(C,self.rho_Rx))
        c_w = self.gamma_w / L_w
        beta_w = (1 - self.gamma_w) * torch.sqrt(C0 * self.nu * (1 - self.nu) * L_w_prev / L_w)
        W_prev = W.clone()
        W_prev = W_prev.to(self.device)
        #print(W.device)	

        for _ in range(10):
            
            W_tilde = W + beta_w * (W - W_prev)
            grad_W = grad_H_W(W_tilde, C, self.Rx)
            W_bar = W_tilde - c_w * grad_W
            W_prev = W.clone()
            W = prox_f(W_bar, c_w)

        C_prev = C.clone()
        beta_c = torch.sqrt(C0 * self.nu * (1 - self.nu))
        C_tilde = C + beta_c * (C - C_prev)
        grad_C = grad_H_C_reg(W, C_tilde, self.Rx, alpha)
        C_bar = C_tilde - c_c * grad_C
        C_prev = C.clone()
        C = prox_g(C_bar, c_c, self.eps)

        return W, C, L_w




# Define the TitanLayer
class TitanLayer(nn.Module):
    def __init__(self, Rx, rho_Rx, gamma_c, gamma_w, eps, nu, input_dim, zeta):
        super(TitanLayer, self).__init__()
        self.alpha_net = AlphaNetwork(input_dim)
        self.deterministic_block = DeterministicBlock(Rx, rho_Rx, gamma_c, gamma_w, eps, nu, zeta)

    def forward(self, W, C, L_w_prev):
        alpha = self.alpha_net(W, C)
        W, C, L_w = self.deterministic_block(W, C, alpha, L_w_prev)
        return W, C, L_w, alpha
    




class TitanIVAGNet(nn.Module):
    def __init__(self, input_dim, num_layers=20, gamma_c=1, gamma_w=0.99, eps=1e-12, nu=0.5, zeta=0.1):
        super(TitanIVAGNet, self).__init__()
        self.num_layers = num_layers
        self.gamma_c = torch.tensor(gamma_c)
        self.gamma_w = torch.tensor(gamma_w)
        self.eps = torch.tensor(eps)
        self.nu = torch.tensor(nu)  
        self.zeta = torch.tensor(zeta)
        self.alpha_network = AlphaNetwork(input_dim)
        self.alphas = [torch.FloatTensor([1]).to('cuda') for _ in range(num_layers)]
        self.input_dim = input_dim
        self.layers = nn.ModuleList([
            TitanLayer(None, None, gamma_c, gamma_w, eps, nu, input_dim, zeta)
            for _ in range(num_layers)
        ])
    


    def initialize_L_w(self, C, rho_Rx, K):
        l_sup = max((self.gamma_w * self.alphas[0]) / (1 - self.gamma_w), rho_Rx * 2 * K * (1 + torch.sqrt(2 / (self.alphas[0] * self.gamma_c))))
        C0 = min(self.gamma_c**2 / K**2, (self.alphas[0] * self.gamma_w / ((1 + self.zeta) * (1 - self.gamma_w) * l_sup)), (rho_Rx / ((1 + self.zeta) * l_sup)))
        l_inf = (1 + self.zeta) * C0 * l_sup
        return max(l_inf, lipschitz(C, rho_Rx))
    
    
    def forward(self, X, A):
        N,_,K = X.shape
        input_dim = N * N * K + K * K * N
        Rx = cov_X(X)
        rho_Rx = spectral_norm_extracted(Rx, K, N)

        
        W, C = initialize(N, K, init_method='random', Winit=None, Cinit=None, X=X, Rx=Rx, seed=None)
        
        L_w_prev = self.initialize_L_w(C, rho_Rx, K)


        for i, layer in enumerate(self.layers):
            layer.alpha_net = AlphaNetwork(input_dim)  # Ensure each layer has its own alpha_net
            layer.deterministic_block = DeterministicBlock(Rx, rho_Rx, self.gamma_c, self.gamma_w, self.eps, self.nu, self.zeta)  # Ensure each layer has its own deterministic block
            W, C, L_w, alpha = layer(W, C, L_w_prev)
            L_w_prev = L_w
            self.alphas[i] = alpha.item()
            

        
        isi_score = joint_isi(W, A)

        return W, C, isi_score

