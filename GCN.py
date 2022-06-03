
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H, A):
        W = self.weight
        b = self.bias

        HW = torch.mm(H, W)
        # AHW = SparseMM.apply(A, HW)
        AHW = torch.spmm(A, HW)
        if self.bias is not None:
            return AHW + b
        else:
            return AHW

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(torch.nn.Module):

    def __init__(self, gl, ll, dropout):
        super(GCN, self).__init__()
        if ll[0] != gl[-1]:
            assert 'Graph Conv Last layer and Linear first layer sizes dont match'
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.graphlayers = nn.ModuleList([GraphConvolution(gl[i], gl[i+1], bias=True) for i in range(len(gl)-1)])
        self.linlayers = nn.ModuleList([nn.Linear(ll[i], ll[i+1]) for i in range(len(ll)-1)])

    def forward(self, H, A):
        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        for idx, hidden in enumerate(self.graphlayers):
            H = F.relu(hidden(H,A))
            if idx < len(self.graphlayers) - 2:
                H = F.dropout(H, self.dropout, training=self.training)

        H_emb = H

        for idx, hidden in enumerate(self.linlayers):
            H = F.relu(hidden(H))

        # print(H)
        return F.softmax(H, dim=1)

    def __repr__(self):
        return str([self.graphlayers[i] for i in range(len(self.graphlayers))] + [self.linlayers[i] for i in range(len(self.linlayers))])


class CutLoss(torch.autograd.Function):
    '''
    Class for forward and backward pass for the loss function described in https://arxiv.org/abs/1903.00614
    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix
    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''

    @staticmethod
    def forward(ctx, Y, A):
        ctx.save_for_backward(Y,A)
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        YbyGamma = torch.div(Y, Gamma.t())
        # print(Gamma)
        Y_t = (1 - Y).t()
        loss = torch.tensor([0.], requires_grad=True).to('cuda')
        idx = A._indices()
        data = A._values()
        for i in range(idx.shape[1]):
            # print(YbyGamma[idx[0,i],:].dtype)
            # print(Y_t[:,idx[1,i]].dtype)
            # print(torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i])
            loss += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i]
            # print(loss)
        # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        Y, A, = ctx.saved_tensors
        idx = A._indices()
        data = A._values()
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        # print(Gamma.shape)
        gradient = torch.zeros_like(Y)
        # print(gradient.shape)
        for i in range(gradient.shape[0]):
            for j in range(gradient.shape[1]):
                alpha_ind = (idx[0, :] == i).nonzero()
                alpha = idx[1, alpha_ind]
                A_i_alpha = data[alpha_ind]
                temp = A_i_alpha / torch.pow(Gamma[j], 2) * (Gamma[j] * (1 - 2 * Y[alpha, j]) - D[i] * (
                            Y[i, j] * (1 - Y[alpha, j]) + (1 - Y[i, j]) * (Y[alpha, j])))
                gradient[i, j] = torch.sum(temp)

                l_idx = list(idx.t())
                l2 = []
                l2_val = []
                # [l2.append(mem) for mem in l_idx if((mem[0] != i).item() and (mem[1] != i).item())]
                for ptr, mem in enumerate(l_idx):
                    if ((mem[0] != i).item() and (mem[1] != i).item()):
                        l2.append(mem)
                        l2_val.append(data[ptr])
                extra_gradient = 0
                if (l2 != []):
                    for val, mem in zip(l2_val, l2):
                        extra_gradient += (-D[i] * torch.sum(
                            Y[mem[0], j] * (1 - Y[mem[1], j]) / torch.pow(Gamma[j], 2))) * val
                print(f"i: {i} and j: {j} from {gradient.shape[0]} and {gradient.shape[0]}")
                gradient[i, j] += extra_gradient

        # print(gradient)
        return gradient, None

def test_partition(Y):
    _, idx = torch.max(Y, 1)
    return idx

def symnormalise(M):
    """
    symmetrically normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def custom_loss_equalpart(Y, A, beta):
    '''
    loss function described in https://arxiv.org/abs/1903.00614
    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : dense adjecency matrix
    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A + beta* (I^T Y - n/g)^2
    '''
    # beta = 0.0001
    D = torch.sum(A, dim=1)
    n = Y.shape[0]
    g = Y.shape[1]
    Gamma = torch.mm(Y.t(), D.unsqueeze(1))
    # print(F.softmax(Y/0.1, 1)[0:10,:])
    # print(Y[0:10,:])
    # print(torch.mm(torch.ones(1,n).to('cuda')/n, F.softmax(Y/0.1, 1)))
    # balance_loss = torch.sum(torch.pow(torch.mm(torch.ones(1,n).to('cuda')/n, F.softmax((Y + torch.randn(Y.shape).to('cuda') * 0.2)/0.1, 1)) - 1/g , 2))
    # print("Ones", torch.ones(1, n))
    # print("Y", Y)
    # print("Sum", torch.sum(Y, dim=1))
    # print("Balance:", torch.mm(torch.ones(1, n).to('cuda'), Y))
    # input()
    # balance_loss = torch.sum(torch.pow(torch.mm(torch.ones(1, n).to('cuda'), Y) - (n / g), 2))
    balance_loss = torch.sum(torch.pow(torch.sum(Y, dim=0) - (n / g), 2))
    partition_loss = torch.sum(torch.mm(torch.div(Y.float(), Gamma.t()), (1 - Y).t().float()) * A.float())
    print('Partition Loss:{0:.3f} \t Balance Loss:{1:.3f}'.format(partition_loss, balance_loss))
    loss = partition_loss + beta * balance_loss
    return loss


def get_stats(node_idx, n_part):
    bucket = torch.zeros(n_part)
    for i in range(n_part):
        # print(i)
        bucket[i] = torch.sum((node_idx == i).int())

    imbalance = torch.mean(torch.abs(bucket-len(node_idx)/2) * 100/len(node_idx))
    # imbalance = torch.mean(torch.pow(bucket/len(node_idx) - 1/n_part,2)) * 100
    if n_part == 2:
        print('Total Elements: {} \t Partition 1: {} \t Partition 2: {}'.format(len(node_idx), bucket[0], bucket[1]))
    if n_part == 3:
        print('Total Elements: {} \t Partition 1: {} \t Partition 2: {} \t Partition 3: {}'.format(len(node_idx), bucket[0], bucket[1], bucket[2]))

    print('Imbalance = {0:.3f}%'.format(imbalance))
    return imbalance


def get_edgecut(As, node_idx):
    idx = As.coalesce().indices()
    values = As.coalesce().values()
    different_part = (node_idx[idx[0,:]] ^ node_idx[idx[1,:]]).type(torch.cuda.FloatTensor)
    edgecut = torch.sum(different_part * values) / 2
    totalwt = torch.sum(values)/2
    print('Edgecut = {0:.3f} \t total edge weight = {1:.3f} \t percent of edge weight cut = {2:.3f}%'.format(edgecut, totalwt, edgecut*100/totalwt))
    return edgecut*100/totalwt


class HypEdgeLst(object):
    '''
    This handles parsing and calculating hyperedges cut
    '''
    def __init__(self, filename):
        file = open(filename, 'r')
        self.file_name = filename
        self.hyedge = []
        num_hyedges = 0
        nodes = set()
        for idx, line in enumerate(file):
            element = line.strip().split()[0:2]
            # if idx == 0:
            #     self.num_hyedges = int(element[0])
            #     self.num_nodes = int(element[1])
            # else:
            hyedge = np.asarray(list(map(int, element))) - 1
            self.hyedge.append(hyedge)
            num_hyedges += 1
            nodes.add(element[0])
            nodes.add(element[1])
        self.num_hyedges = num_hyedges
        self.num_nodes = len(nodes)

    def get_cut(self, node_idx):
        int_hyedge = 0
        for hyedge in self.hyedge:
            node_class = node_idx[hyedge].data.cpu().numpy()
            # print("Node_idx", node_idx)
            # print("node[hyedge]", node_idx[hyedge])
            # print("hyedge", hyedge)
            # print("nodeclass", node_class)
            # input()
            # if not np.all(node_class == node_class[0]):
            if node_class[0] != node_class[1]:
                # print('cut')
                int_hyedge += 1

        print('Number of Hyper Edges intersected = {}/{} = {}'.format(int_hyedge, self.num_hyedges, int_hyedge/self.num_hyedges))
        return int_hyedge

