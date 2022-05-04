from ast import walk
from collections import defaultdict
import networkx as nx
from karateclub import DeepWalk
import numpy as np
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

    model = DeepWalk(walk_length=40, walk_number=30, dimensions=64, window_size=10, workers=8)
    model.fit(G)
    features = model.get_embedding()

    end = time.time()
    print(f"Feature creation took {end - start} seconds in a graph with {len(G)} nodes.")
    print(f"A matrix of {features.shape} was created")

    return features


def create_feature_files(prefix, graph_file_names):
    for file in graph_file_names:
        adj_list = load_graph_file(prefix + file)
        new_list = to_zero_based(adj_list)
        features = create_features(new_list)
        feature_location = "feature_files/" + prefix + file
        np.save(feature_location, features)
        print(f"Features saved into the file: {feature_location}")
        print("################################################################")


def read_feature_file(file_name):
    features = np.load("feature_files/" + file_name + ".npy")
    print(f"Features file {file_name} readed. Returned a {features.shape} array")

if __name__ == "__main__":
    small_graphs_prefix = "small_graphs/"
    medium_graphs_prefix = "medium_graphs/"
    large_graphs_prefix = "large_graphs/"

    small_graphs = ["add20", "data", "3elt", "uk", "add32", "bcsstk33", "whitaker3", "crack"]
    medium_graphs = ["fe_body", "t60k", "wing", "finan512"]
    large_graphs = ["fe_rotor", "598a", "m14b", "auto"]

    test_graphs = ["whitaker3", "crack"]

    prefixes = [small_graphs_prefix,
                medium_graphs_prefix,
                large_graphs_prefix]
    graph_files = [small_graphs,
                medium_graphs,
                large_graphs]

    for i, file in enumerate(graph_files):
        create_feature_files(prefixes[i], file)

    #for path in test_graphs:
    #    read_feature_file(small_graphs_prefix + path)
