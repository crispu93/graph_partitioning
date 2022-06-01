import networkx as nx
from graph_utils import load_graph_file, to_zero_based, load_edge_list_file
import nxmetis
from karateclub import DeepWalk
import time
import numpy as np

class GraphPartitioning:
    def __init__(self, prefix_name, name, num_parts, vol=False, chaco=True):
        self.prefix_name = prefix_name
        self.name = name
        self.file_name = prefix_name + name
        self.vol = vol
        self.num_parts = num_parts
        if chaco:
            self._create_chaco_graph()
        else:
            self._create_edge_graph()

    
    def _create_chaco_graph(self):
        adj_list = load_graph_file(self.file_name)
        adj_list = to_zero_based(adj_list)
        self.G = nx.from_dict_of_lists(adj_list)
    
    def _create_edge_graph(self):
        adj_list = load_edge_list_file(self.file_name)
        adj_list = to_zero_based(adj_list)
        self.G = nx.from_dict_of_lists(adj_list)
    
    def to_metis_partition(self):
        """Partitions a graph using networkx-metis lib"""

        if self.vol == False:
            options = nxmetis.MetisOptions(contig=False, ncuts=3, objtype=0)
            (cutcost, parts) = nxmetis.partition(self.G, nparts=self.num_parts, options=options)
            print(
                f"The Graph {file_name} has been partitioned with cut={cutcost}"
                )
        else:
            options = nxmetis.MetisOptions(contig=False, ncuts=3, objtype=1)
            (cutcost, parts) = nxmetis.partition(self.G, nparts=self.num_parts, options=options)
            print(
                f"The Graph {file_name} has been partitioned according to vol with cut={cutcost}"
                )

        for i,p in enumerate(parts):
            print(f"Lenght of partition #{i}:{len(p)}")

        self.cutcost = cutcost
        self.parts = parts

    def draw_partition(self):
        """Draw a partition using networkx-metis lib"""

        colors = ['red', 'blue', 'green', 'yellow']
        for i, p in enumerate(self.parts):
            for j in p:
                self.G.nodes[j]['color'] = colors[i]

        print("Drawing the partitioned graph")
        A = nx.nx_agraph.to_agraph(self.G)
        A.node_attr['style']='filled'
        A.draw("images/" + self.file_name + ".png", prog="sfdp")
        print(
            f"Image succesfully saved to {'images/' + self.file_name + '.png'}"
            )

    def create_features(self):
        start = time.time()

        model = DeepWalk(walk_length=80, walk_number=60, dimensions=64, window_size=15, workers=8)
        model.fit(self.G)
        features = model.get_embedding()

        end = time.time()
        print(f"Feature creation took {end - start} seconds in a graph with {len(self.G)} nodes.")
        print(f"A matrix of {features.shape} was created")

        # self.features = features
        feature_location = "feature_files/" + self.file_name
        np.save(feature_location, features)

    def read_feature_file(self):
        self.features = np.load("feature_files/" + self.file_name + ".npy")
        print(f"Features file {self.file_name} readed. Returned a {self.features.shape} array")
        return self.features

if __name__ == "__main__":
    small_graphs_prefix = "small_graphs/"
    medium_graphs_prefix = "medium_graphs/"
    large_graphs_prefix = "large_graphs/"

    small_graphs = ["bcsstk33", "whitaker3", "crack"]
    medium_graphs = ["fe_body", "t60k", "wing", "finan512"]
    large_graphs = ["fe_rotor", "598a", "m14b", "auto"]

    for file_name in small_graphs:
        metis_part = GraphPartitioning(small_graphs_prefix, file_name, 4)
        metis_part.to_metis_partition()
        metis_part.draw_partition()

    # for file_name in large_graphs:
    #     metis_part = GraphPartitioning(large_graphs_prefix, file_name, 4, vol=True)
    #     # metis_part.to_metis_partition()
    #     # metis_part.draw_partition()
    #     metis_part.create_features()
    #     del metis_part