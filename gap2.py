
from __future__ import division
from __future__ import print_function
import tqdm
import time
import argparse
import numpy as np
import scipy.sparse as sp
#from scipy import sparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from GCN import (CutLoss, test_partition, HypEdgeLst,
sparse_mx_to_torch_sparse_tensor, symnormalise, 
GCN, custom_loss_equalpart, get_edgecut, get_stats)
import networkx as nx
from graph_partitioning import GraphPartitioning
import matplotlib.pyplot as plt

SMALL_GRAPHS_PREFIX = "small_graphs/"
MEDIUM_GRAPHS_PREFIX = "medium_graphs/"
LARGE_GRAPHS_PREFIX = "large_graphs/"

SMALL_GRAPHS = ["add20", "bcsstk33", "whitaker3", "crack"]
MEDIUM_GRAPHS = ["fe_body", "t60k", "wing", "finan512"]
LARGE_GRAPHS = ["fe_rotor", "598a", "m14b", "auto"]

device = None

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print('Using device', device_id, torch.cuda.get_device_name(device_id))
    device = torch.device("cuda")
    print('DEVICE:', device)
else:
    print("WARNING: You don't have a CUDA device")
    device = torch.device("cpu")
device = torch.device("cpu")


# def Train(model, x, adj, A, optimizer):
#     '''
#     Training Specifications
#     '''

#     max_epochs = 5
#     min_loss = 100
#     for epoch in (range(max_epochs)):
#         Y = model(x, adj)
#         print(Y)
#         loss = CutLoss.apply(Y, A)
#         # loss = custom_loss(Y, A)
#         print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))
#         if loss < min_loss:
#             min_loss = loss.item()
#             torch.save(model.state_dict(), "./trial_weights.pt")
#         loss.backward()
#         optimizer.step()


# def Test(model, x, adj, A, *argv):
#     '''
#     Test Final Results
#     '''
#     model.load_state_dict(torch.load("./trial_weights.pt"))
#     Y = model(x, adj)
#     print(Y)
#     node_idx = test_partition(Y)
#     print(node_idx)
#     # if argv != ():
#     #     if argv[0] == 'debug':
#     #         print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss(Y,A).item()))
#     # else:
#     print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(CutLoss.apply(Y,A).item()))


def Train_dense(model, x, adj, A, As, optimizer, beta0, hyedge_lst = None, args = None):
    '''
    Training Specifications
    '''

    max_epochs = 100
    min_loss = 100
    min_cut = 10000000000
    # beta = 0
    Hcut_arr = []
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.2)
    for epoch in (range(max_epochs)):
        # beta = beta + beta0/max_epochs
        Y = model(x, adj)
        loss = custom_loss_equalpart(Y, A, beta0)
        print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))

        loss.backward()
        optimizer.step()
        
        cut, imbalance, edge_cut = test_epoch(model, x, adj, hyedge_lst, As, args)
        print("Fraction: ", cut/hyedge_lst.num_hyedges)
        Hcut_arr.append(cut/hyedge_lst.num_hyedges)
        graph_name = hyedge_lst.file_name.split("/")[-1]
        if cut <= min_cut:
            if args == 2:
                if imbalance < 15: # This is to ensure one partition is not 0
                    min_cut = cut
                    torch.save(model.state_dict(), "model_weights/"+graph_name+'_'+str(args)+"_MinCut.pt")
            else:
                min_cut = cut
                torch.save(model.state_dict(),
                           "model_weights/"+graph_name+'_'+str(args)+"_MinCut.pt")

        # scheduler.step()
        print('======================================')
    plt.plot(Hcut_arr)
    plt.ylabel('Fraction of HyperEdge Cut')
    plt.xlabel('Epochs')
    plt.title(graph_name)
    plt.show()


def test_epoch(model, x, adj, hyedge_lst, As, args):
    '''
    Check for the the number of hyperedges cut after each epoch
    '''
    Y = model(x, adj)
    node_idx = test_partition(Y)
    imbalance = get_stats(node_idx, args)
    cut = hyedge_lst.get_cut(node_idx)
    edge_cut = get_edgecut(As, node_idx)
    return cut, imbalance, edge_cut


def Test_dense(model, x, adj, A, As, beta, hyedge_lst, args):
    '''
     Test Final Results
     '''
    print('===== Hyper Edge Cut Minimization =====')
    graph_name = hyedge_lst.file_name.split("/")[-1]
    model.load_state_dict(torch.load("model_weights/"+graph_name+'_'+str(args)+"_MinCut.pt"))
    Y = model(x, adj)
    node_idx = test_partition(Y)
    get_stats(node_idx, args)
    hyedge_lst.get_cut(node_idx)
    get_edgecut(As, node_idx)

    return node_idx


def dense_test_and_train(model, x, adj, A, As, optimizer, beta, hyedge_lst, args):
    '''
    Training and Testing combined into a single code to be called
    '''
    #Train
    Train_dense(model, x, adj, A, As, optimizer, beta, hyedge_lst, args)

    # Test the best partition
    print('#####Best Result#####')
    node_idx = Test_dense(model, x, adj, A, As, beta, hyedge_lst, args)
    return node_idx


def main():
    '''
    Adjecency matrix and modifications
    '''
    # Datasets creation
    # file_name = "aves-thornbill-farine"
    # file_name = "test_small_graph"
    file_name = "insecta-ant-colony1-day01"
    num_partitions = 2
    graph_part = GraphPartitioning(SMALL_GRAPHS_PREFIX, file_name, num_partitions, vol=True, chaco=False)

    # graph_part = GraphPartitioning(SMALL_GRAPHS_PREFIX, file_name, num_partitions, vol=True, chaco=True)
    # adj_list = list(map(list, iter(graph_part.G.adj.values())))
    graph_part.create_features()

    A = nx.to_numpy_matrix(graph_part.G)
    A = sp.csr_matrix(A)
    #A = input_matrix()

    # Modifications
    A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to('cuda') # SciPy to Torch sparse
    As = sparse_mx_to_torch_sparse_tensor(A).to('cuda')  # SciPy to sparse Tensor
    A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to('cuda')   # SciPy to Torch Tensor
    print(A)

    '''
    Declare Input Size and Tensor
    '''
    # N = A.shape[0]
    # d = 512

    # torch.manual_seed(100)
    # x = torch.randn(N, d)
    # x = x.to('cuda')
    x = torch.Tensor(graph_part.read_feature_file()).to('cuda')

    '''
    Model Definition
    '''
    gl = [x.size(1), 512, 64]
    ll = [64, 2]

    model = GCN(gl, ll, dropout=0.5).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6, betas=(0.5, 0.99))
    print(model)

    # check_grad(model, x, adj, A, As)
    filename = "graph_files/" + graph_part.file_name + ".edges"
    hyedge_lst = HypEdgeLst(filename)
    #Train
    #beta0 = 0.0005
    beta0 = 0.01
    args = num_partitions
    dense_test_and_train(model, x, adj, A, As, optimizer, beta0, hyedge_lst, args)
    # Train_dense(model, x, adj, A, As, optimizer, beta0, hyedge_lst, num_partitions)
    # Train(model, x, adj, As, optimizer)
    graph_part.to_metis_partition()
    graph_part.part_graph()

    # graph_part.draw_graph()
    # graph_part.draw_partition()
    # Test the best partition

    # Test(model, x, adj, As)

if __name__ == '__main__':
    main()

