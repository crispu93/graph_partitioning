from ast import walk
from collections import defaultdict
import networkx as nx
from karateclub import DeepWalk
import time

def load_graph_file(file_name):
    """Represents the graph in the file into an adjacency list"""

    path = "graph_files/" + file_name + ".graph"
    adj_list = defaultdict(list)
    with open(path) as fp:
        fp.readline()
        for i, line in enumerate(fp):
            adj_nodes = [int(x) for x in line.strip().split()]
            adj_list[i+1].extend(adj_nodes)
            #print(i+1, ":", adj_list[i+1])
    return adj_list

def to_zero_based(adj_list):
    new_adj_list = defaultdict(list)

    for key, value in adj_list.items():
        adj_nodes = [x-1 for x in value]
        new_adj_list[key-1] = adj_nodes

    return new_adj_list


def create_features(adj_list):
    start = time.time()
    G = nx.from_dict_of_lists(adj_list)

    model = DeepWalk(walk_length=60, walk_number=60, dimensions=128, window_size=15, workers=8)
    model.fit(G)
    features = model.get_embedding()

    end = time.time()
    print(f"Feature creation took {end - start} seconds in a graph with {len(G)} nodes.")
    print(f"A matrix of {features.shape} was created")

    return features

if __name__ == "__main__":
    #adj_list = load_graph_file("small_graphs/add20")
    adj_list = load_graph_file("t60k")
    new_list = to_zero_based(adj_list)
    create_features(new_list)
