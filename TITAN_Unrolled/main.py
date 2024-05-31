import torch
import torch.optim as optim
from model import TitanIVAGNet
from data import *
from tools import *
from torch.utils.data import DataLoader

# Define the number of epochs and other parameters
num_epochs = 100
T = 1000
K = 2
N = 3

input_dim = N * N * K + K * K * N 
num_layers = 20

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]

metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D']




gamma_c = 1
gamma_w = 0.99
eps = 1e-12
nu = 0.5
size = 100
zeta = 0.1 


# créer le dataset
dataset = MonDataset(T, K, N, metaparameters_multiparam, size = size)

# créer le dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



# Now, when creating the optimizer, collect parameters from all layers
model = TitanIVAGNet(input_dim, num_layers=num_layers, gamma_c=gamma_c, gamma_w=gamma_w, eps=eps, nu=nu, zeta=zeta).cuda()

# Collect parameters from all alpha networks
alpha_net_params = []
for layer in model.layers:
    alpha_net_params.extend(layer.alpha_net.parameters())


optimizer = optim.Adam(alpha_net_params, lr=0.001)
# Training loop
for epoch in range(num_epochs):
    for X, A in dataloader:
        X = X[0]
        A = A[0]
        
        #print(f"Batch X shape: {X.shape}, Batch A shape: {A.shape}")

        # Forward pass to compute ISI scores and update parameters
        model.train()
        optimizer.zero_grad()
        W, C, _ = model(X, A)
        isi_score = joint_isi(W, A)
        print(f"ISI score before backward: {isi_score.item()}")
        isi_score.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, ISI Score: {isi_score.item()}")
    """     
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data) """


# After training, you can access the trained model parameters, including the optimized alpha values, using model.parameters().





