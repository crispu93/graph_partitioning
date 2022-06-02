from platform import node
from graph_partitioning import GraphPartitioning
from graph_sage import CutLoss, GraphSage, PartitioningModule
from utils import test_partition
import torch
import torch.optim as optim
import torch.nn  as nn
import random
import numpy as np
from graph_utils import read_feature_file
from graph_sage import UnsupervisedLoss
import sys

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


def apply_model(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, min_loss):#, learn_method, batch_size):
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
    
    optimizer = torch.optim.SGD(params, lr=0.7)
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
    loss_cut = CutLoss.apply(Y, graphSage.adj_lists)

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


def Train(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, max_epochs, min_loss):
    """G is a networkx graph"""

    min_loss = 100
    for epoch in range(max_epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        graphSage, graphPartitioner = apply_model(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, min_loss)


def Test(graphSage, graphPartitioner):
    '''
    Test Final Results
    '''
    graphSage.load_state_dict(torch.load("./emb_trial_weights.pt"))
    graphPartitioner.load_state_dict(torch.load("./part_trial_weights.pt"))
    node_batch = list(range(len(adj_list)))
    embs_batch = graphSage(node_batch)
    Y = graphPartitioner(embs_batch)
    node_idx = test_partition(Y)
    print("node_idx", node_idx)
    # if argv != ():
    #     if argv[0] == 'debug':
    #         print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss(Y,A).item()))
    # else:
    print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(CutLoss.apply(Y, graphSage.adj_lists).item()))


def main():
    # Datasets creation
    file_name = "aves-thornbill-farine"
    num_partitions = 2
    graph_part = GraphPartitioning(SMALL_GRAPHS_PREFIX, file_name, num_partitions, vol=True, chaco=False)
    graph_part.create_features()

    # Parameters for GraphSAGE
    num_layers = 5
    hidden_emb_size = 128
    features = torch.Tensor(graph_part.read_feature_file()).to(device)
    adj_list = list(map(list, iter(graph_part.G.adj.values())))

    # Creates graphSage model
    graphSage = GraphSage(num_layers, features.size(1), hidden_emb_size, features, adj_list, device, gcn=False, agg_func='MEAN')
    graphSage = graphSage.to(device)

    # Instance for the graph partitioning module
    ll = [hidden_emb_size, 64, 64, graph_part.num_parts]
    graphPartitioner = PartitioningModule(ll)
    graphPartitioner = graphPartitioner.to(device)
    
    # Parameters to train
    # Unsupervised loss for graphSage model
    node_batch = list(range(len(adj_list)))
    unsupervised_loss = UnsupervisedLoss(adj_list, node_batch, device)
    unsup_loss_type = 'margin' # can be 'normal' for more samples
    max_epochs = 10
    min_loss = 100

    # Train
    Train(graphSage, graphPartitioner, unsupervised_loss, unsup_loss_type, max_epochs, min_loss)

    # Test the best partition
    Test(graphSage, graphPartitioner)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    main()