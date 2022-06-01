import torch
from graph_utils import load_graph_file, to_zero_based, load_edge_list_file
import networkx as nx
print(torch.cuda.is_available())

# http://www.trustlet.org/datasets/
small_graphs_prefix = "small_graphs/"
paths = ["aves-songbird-social", "aves-thornbill-farine", "insecta-ant-colony1-day01", "insecta-ant-colony1-day12", "insecta-ant-colony4"]
for path in paths:
    adj_list = load_edge_list_file(small_graphs_prefix + path)
    G = nx.from_dict_of_lists(adj_list)
    # print(adj_list)
    print("Nodes: ", G.number_of_nodes())
# G = nx.Graph(nx.nx_pydot.read_dot(path))
# H = nx.convert_node_labels_to_integers(G)
# adj_list = list(map(list, iter(H.adj.values())))

#print(adj_list)
