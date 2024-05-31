import torch


def sym(A):
    if A.ndim == 2:
        return (A.transpose(0, 1) + A) / 2
    else:
        return (A + torch.moveaxis(A, 0, 1)) / 2


def cov_X(X):
    _, T, _ = X.size()
    Rx = torch.einsum('NTK,MTJ->KJNM', X, X) / T
    return Rx


def spectral_norm(M):
    if M.dim() == 2:
        return torch.linalg.norm(M, ord=2)
    else:
        return torch.max(torch.linalg.norm(M, ord=2,dim=(0,1)))
    

def spectral_norm_extracted(Rx,K,N):
    return torch.max(torch.norm(torch.reshape(Rx,(K,K*N,N)),p=2,dim=(1,2)))


def smallest_singular_value(C):
    _, s, _ = torch.svd(C.permute(2, 0, 1))
    return torch.min(s)

def block_diag(W):
    N, N, K = W.size()
    W_bd = torch.zeros(K, K*N, N,device='cuda')
    for k in range(K):
        W_bd[k, k*N:(k+1)*N, :] = W[:, :, k].t()
    return W_bd

def blocks_to_full(W_blocks, K, N):
    W_full = torch.zeros(N, N, K,device='cuda')
    for k, W_k in enumerate(W_blocks):
        begin = 0
        for W_kl in W_k:
            n_l = W_kl.size(0)
            W_full[begin:begin+n_l, begin:begin+n_l, k] = W_kl
            begin += n_l
    return W_full

def full_to_blocks(W_full, idx_W, K):
    W_blocks = []
    for k in range(K):
        W_k = []
        L_k = len(idx_W[k])
        for l in range(L_k):
            W_kl = W_full[idx_W[k][l], idx_W[k][l], k]
            W_k.append(W_kl)
        W_blocks.append(W_k)
    return W_blocks

def lipschitz(C,lam):
    return spectral_norm(C)*lam

def joint_isi(W, A):
    N, _, _ = W.size()
    G_bar = torch.sum(torch.abs(torch.einsum('nNk,Nvk->nvk', W, A)), dim=2)
    score = (torch.sum(torch.sum(G_bar / torch.max(G_bar, dim=0)[0], dim=0) - 1) +
             torch.sum(torch.sum(G_bar.t() / torch.max(G_bar.t(), dim=0)[0], dim=0) - 1))
    return score / (2 * N * (N - 1))

def decrease(cost, verbose=0):
    accr = torch.tensor(cost[:-1]) - torch.tensor(cost[1:])
    if torch.all(accr >= 0):
        return True
    else:
        if verbose >= 1:
            for i in range(len(accr)):
                if accr[i] < 0:
                    print("increase at index :", i)
                    print("an increase of :", -accr[i])
                    break
        return False
    
def diff_criteria(A, B, mode='full'):
    if mode == 'full':
        if A.shape != B.shape:
            raise ValueError("A and B must be of the same dimension")
        elif A.ndim < 2 or A.ndim > 3:
            raise ValueError("Only tensors of order 2 or 3 are accepted")
        res = 0
        D = A - B

        max_norm = torch.max(torch.sum(D ** 2, dim=1))
        return max_norm / (2 * A.size(0))


    elif mode == 'blocks':
        res = 0
        if len(A) != len(B):
            raise ValueError("A and B must be of the same dimension")
        for k, A_k in enumerate(A):
            B_k = B[k]
            if len(A_k) != len(B_k):
                raise ValueError("A_k and B_k must have the same blocks")
            for l, A_kl in enumerate(A_k):
                B_kl = B_k[l]
                if A_kl.shape != B_kl.shape:
                    raise ValueError("A_k and B_k must have the same blocks")
                N_kl, _ = A_kl.shape
                for n in range(N_kl):
                    res = max(res, torch.dot(A_kl[n, :], B_kl[n, :]) / (2 * N_kl))
        return res

        

