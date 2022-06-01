from graph_partitioning import GraphPartitioning
from graph_sage import CutLoss, GraphSage
from utils import test_partition
import torch
import torch.optim as optim
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

def Train(model, adj_list, optimizer, unsupervised_loss, unsup_loss):
    """G is a networkx graph"""

    """ For graphSAGE"""
    num_neg = None
    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
    optimizer.zero_grad()
    model.zero_grad()

    visited_nodes = set()
    max_epochs = 5
    min_loss = 100
    node_batch = list(range(len(adj_list)))

    for epoch in range(max_epochs):
		# extend nodes batch for unspervised learning
		# no conflicts with supervised learning
        node_batch = np.asarray(list(unsupervised_loss.extend_nodes(node_batch, num_neg=num_neg)))
        visited_nodes |= set(node_batch)

		# feed nodes batch to the graphSAGE
		# returning the nodes embeddings
        Y, embs_batch = model(node_batch)

        if unsup_loss == 'margin':
            loss_net = unsupervised_loss.get_loss_margin(embs_batch, node_batch)
        elif unsup_loss == 'normal':
            loss_net = unsupervised_loss.get_loss_sage(embs_batch, node_batch)
        loss = loss_net

        #Y = model(adj_list)
        Y = model(node_batch)
        loss = CutLoss.apply(Y, adj_list)
        print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))
        if loss < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "./trial_weights.pt")
        loss.backward()
        optimizer.step()


def Test(model, adj_list):
    '''
    Test Final Results
    '''
    model.load_state_dict(torch.load("./trial_weights.pt"))
    node_batch = list(range(len(adj_list)))
    Y = model(node_batch)
    node_idx = test_partition(Y)
    print("node_idx", node_idx)
    # if argv != ():
    #     if argv[0] == 'debug':
    #         print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss(Y,A).item()))
    # else:
    print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(CutLoss.apply(Y, adj_list).item()))


def main():
    # file_name = SMALL_GRAPHS[0]
    file_name = "aves-thornbill-farine"
    num_partitions = 2
    graph_part = GraphPartitioning(SMALL_GRAPHS_PREFIX, file_name, 2, vol=True, chaco=False)
    graph_part.create_features()
    num_layers = 3
    features = torch.Tensor(graph_part.read_feature_file()).to(device)
    print("features:", features)
    hidden_emb_size = 64
    ll = [64, 64, 64, graph_part.num_parts]
    adj_list = list(map(list, iter(graph_part.G.adj.values())))
    model = GraphSage(num_layers, features.size(1), hidden_emb_size, ll, features, adj_list, device, gcn=False, agg_func='MEAN')
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
    node_batch = list(range(len(adj_list)))
    unsupervised_loss = UnsupervisedLoss(adj_list, node_batch, device)
    unsup_loss = 'margin' # can be 'normal' for more samples

    # Train
    Train(model, adj_list, optimizer, unsupervised_loss, unsup_loss)

    # Test the best partition
    Test(model, adj_list)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    main()