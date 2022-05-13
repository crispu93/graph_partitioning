import torch

def expected_normalized_cut_loss(Y, A):
    '''
    loss function described in https://arxiv.org/abs/1903.00614
    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : dense adjecency matrix
    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    D = torch.sum(A, dim=1)
    Gamma = torch.mm(Y.t(), D.unsqueeze(1))
    # print(Gamma)
    loss = torch.sum(torch.mm(torch.div(Y.float(), Gamma.t()), (1 - Y).t().float()) * A.float())
    return loss

def custom_loss_sparse(Y, A):
    '''
    loss function described in https://arxiv.org/abs/1903.00614
    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix
    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    D = torch.sparse.sum(A, dim=1).to_dense()
    Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
    YbyGamma = torch.div(Y, Gamma.t())
    Y_t = (1 - Y).t()
    loss = torch.tensor([0.])
    idx = A._indices()
    for i in range(idx.shape[1]):
        loss += torch.dot(YbyGamma[idx[0,i],:], Y_t[:,idx[1,i]])
    return loss