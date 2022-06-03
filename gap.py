from collections import defaultdict
from platform import node
from graph_partitioning import GraphPartitioning
from graph_sage import (CutLoss, GraphSage, PartitioningModule, HypEdgeLst,
                        custom_loss_equalpart, get_edgecut, get_stats)
from utils import test_partition
import torch
import torch.optim as optim
import torch.nn  as nn
import random
import numpy as np
from graph_utils import read_feature_file
from graph_sage import UnsupervisedLoss
import sys
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


def Train(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, max_epochs, min_loss, beta):
    """G is a networkx graph"""
    # beta = 0.0005
    min_loss = 100
    for epoch in range(max_epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        graphSage, graphPartitioner = apply_model(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, min_loss, beta)


def Test(graphSage, graphPartitioner, beta):
    '''
    Test Final Results
    '''
    graphSage.load_state_dict(torch.load("./emb_trial_weights.pt"))
    graphPartitioner.load_state_dict(torch.load("./part_trial_weights.pt"))
    node_batch = list(range(len(graphSage.adj_lists)))
    embs_batch = graphSage(node_batch)
    Y = graphPartitioner(embs_batch)
    node_idx = test_partition(Y)
    print("node_idx", node_idx)
    zeros = len([x for x in node_idx if x == 0])
    print("Number of zeros:", zeros)
    get_cut_value(graphSage.adj_lists, node_idx)
    # if argv != ():
    #     if argv[0] == 'debug':
    #         print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss(Y,A).item()))
    # else:
    # print("Normalized cut of the partition", normalized_cut(graphSage.adj_lists, node_idx))
    print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss_equalpart(Y, graphSage.adj_lists, beta).item()))


def apply_model(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, min_loss, beta):#, learn_method, batch_size):
    """ Training embedding and partitioning modules 
        Only supports unsupervised learning for GraphSAGE
    """

    nodes = list(range(len(graphSage.adj_lists)))

    num_neg = None
    if unsup_loss_type == 'margin':
        num_neg = 6
    elif unsup_loss_type == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)
    
    models = [graphSage, graphPartitioner]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)
    
    optimizer = torch.optim.SGD(params, lr=0.007)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    # Note: no support for batch training because training the
    # embedding and partitioning modules at the same time
    # using nodes as node_batch
    visited_nodes = set()

    # extend nodes batch for unspervised learning
    # no conflicts with supervised learning
    nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes, num_neg=num_neg)))
    visited_nodes |= set(nodes_batch)

    # feed nodes batch to the graphSAGE
    # returning the nodes embeddings
    embs_batch = graphSage(nodes_batch)

    # Probabilities matrix
    Y = graphPartitioner(embs_batch)
    # loss_cut = CutLoss.apply(Y, graphSage.adj_lists)
    loss_cut = custom_loss_equalpart(Y, graphSage.adj_lists, beta)


    if unsup_loss_type == 'margin':
        loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
    elif unsup_loss_type == 'normal':
        loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
    loss = loss_net + loss_cut

    if loss < min_loss:
        min_loss = loss.item()
        torch.save(graphSage.state_dict(), "./emb_trial_weights.pt")
        torch.save(graphPartitioner.state_dict(), "./part_trial_weights.pt")

    print('Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(loss.item(), len(visited_nodes), len(nodes)))
    
    loss.backward()

    for model in models:
        nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    return graphSage, graphPartitioner


def dense_test_and_train(graphSage, graphPartitioner, unsupervised_loss, hyedge_lst):
    '''
    Training and Testing combined into a single code to be called
    '''
    unsup_loss_type = 'normal' # can be 'normal' for more samples
    max_epochs = 100
    min_loss = 100
    beta = 0.005
    # Train
    Train_dense(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, max_epochs, min_loss, hyedge_lst, beta)

    # Test the best partition
    node_idx = Test_dense(graphSage, graphPartitioner, hyedge_lst)
    return node_idx


def Train_dense(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, max_epochs, min_loss, hyedge_lst, beta):
    '''
    Training Specifications
    '''
    min_cut = 10000000000
    num_part = graphPartitioner.num_partitions
    # beta = 0
    Hcut_arr = []
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.2)
    for epoch in range(max_epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        graphSage, graphPartitioner = apply_model(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, min_loss, beta)
        cut, imbalance, edge_cut = test_epoch(graphSage, graphPartitioner, hyedge_lst)
        
        Hcut_arr.append(cut/hyedge_lst.num_hyedges)
        graph_name = hyedge_lst.file_name.split("/")[-1]
        if cut <= min_cut:
            if num_part == 2:
                if imbalance < 15: # This is to ensure one partition is not 0
                    min_cut = cut
                    torch.save(graphSage.state_dict(), "model_weights/"+graph_name+'_'+str(num_part)+"_Embbed.pt")
                    torch.save(graphPartitioner.state_dict(), "model_weights/"+graph_name+'_'+str(num_part)+"_Cuts.pt")
            else:
                min_cut = cut
                torch.save(graphSage.state_dict(), "model_weights/"+graph_name+'_'+str(num_part)+"_Embbed.pt")
                torch.save(graphPartitioner.state_dict(), "model_weights/"+graph_name+'_'+str(num_part)+"_Cuts.pt")

        # scheduler.step()
        print('======================================')
    plt.plot(Hcut_arr)
    plt.ylabel('Fraction of HyperEdge Cut')
    plt.xlabel('Epochs')
    plt.title(graph_name)
    plt.show()


def test_epoch(graphSage, graphPartitioner, hyedge_lst):
    '''
    Check for the the number of hyperedges cut after each epoch
    '''
    num_partitions = graphPartitioner.num_partitions
    node_batch = list(range(len(graphSage.adj_lists)))

    embs_batch = graphSage(node_batch)
    Y = graphPartitioner(embs_batch)

    node_idx = test_partition(Y)
    imbalance = get_stats(node_idx, num_partitions)
    cut = hyedge_lst.get_cut(node_idx)
    edge_cut = get_edgecut(graphSage.adj_lists, node_idx)
    return cut, imbalance, edge_cut
    

def Test_dense(graphSage, graphPartitioner, hyedge_lst):
    '''
     Test Final Results
     '''
    print('===== Hyper Edge Cut Minimization =====')
    graph_name = hyedge_lst.file_name.split("/")[-1]
    num_part = graphPartitioner.num_partitions

    graphSage.load_state_dict(torch.load("model_weights/"+graph_name+'_'+str(num_part)+"_Embbed.pt"))
    graphPartitioner.load_state_dict(torch.load("model_weights/"+graph_name+'_'+str(num_part)+"_Cuts.pt"))

    node_batch = list(range(len(graphSage.adj_lists)))
    embs_batch = graphSage(node_batch)
    Y = graphPartitioner(embs_batch)
    node_idx = test_partition(Y)

    get_stats(node_idx, num_part)
    hyedge_lst.get_cut(node_idx)
    get_edgecut(graphSage.adj_lists, node_idx)

    return node_idx


def get_cut_value(adj_list, node_idx):
    edge_cut = 0
    for i, element in enumerate(adj_list):
        for j in element:
            if (node_idx[i] != node_idx[j]):
                edge_cut += 1
    print("Partition Cut:", edge_cut/2)
    return edge_cut / 2


def normalized_cut(adj_list, node_idx):
    edge_cut = 0
    vol = [0]*2
    for i, element in enumerate(adj_list):
        for j in element:
            if (node_idx[i] != node_idx[j]):
                edge_cut += 1
            else:
                vol[node_idx[i]] += 1
                vol[node_idx[j]] += 1
    edge_cut /= 2
    norm_cut = edge_cut/vol[0] + edge_cut/vol[1]

    print("Normalized Cut:", norm_cut)


def main():
    # Datasets creation
    #file_name = "aves-thornbill-farine"
    # file_name = "test_small_graph"
    # file_name = "insecta-ant-colony1-day01"
    # file_name = "bio-CE-LC" Problems, zero index based
    # file_name = "mammalia-voles-rob-trapping"
    file_name = SMALL_GRAPHS[2]
    num_partitions = 2
    #graph_part = GraphPartitioning(SMALL_GRAPHS_PREFIX, file_name, num_partitions, vol=True, chaco=False)
    graph_part = GraphPartitioning(SMALL_GRAPHS_PREFIX, file_name, num_partitions, vol=True, chaco=True)
    adj_list = list(map(list, iter(graph_part.G.adj.values())))
    # graph_part.create_features()
    # graph_part.to_metis_partition()
    # graph_part.draw_partition()

    # Parameters for GraphSAGE
    num_layers = 2
    hidden_emb_size = 256
    features = torch.Tensor(graph_part.read_feature_file()).to(device)
    adj_list = list(map(list, iter(graph_part.G.adj.values())))

    # Creates graphSage model
    graphSage = GraphSage(num_layers, features.size(1), hidden_emb_size, features, adj_list, device, gcn=False, agg_func='MEAN')
    graphSage = graphSage.to(device)

    # Instance for the graph partitioning module
    ll = [hidden_emb_size, 64, graph_part.num_parts]
    graphPartitioner = PartitioningModule(ll)
    graphPartitioner = graphPartitioner.to(device)
    
    # Parameters to train
    # Unsupervised loss for graphSage model
    node_batch = list(range(len(adj_list)))
    unsupervised_loss = UnsupervisedLoss(adj_list, node_batch, device)

    # filename = "graph_files/" + graph_part.file_name + ".edges"
    # hyedge_lst = HypEdgeLst(filename)
    unsup_loss_type = 'normal' # can be 'normal' for more samples
    max_epochs = 20
    min_loss = 100
    beta = 0.01

    # node_idx = dense_test_and_train(graphSage, graphPartitioner, unsupervised_loss, hyedge_lst)
    # print("*"*100)
    # print("Final edge cut:", get_cut_value(adj_list, node_idx))

    # Train
    Train(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, max_epochs, min_loss, beta)

    # Test the best partition
    Test(graphSage, graphPartitioner, beta)

    graph_part.to_metis_partition()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    main()