import networkx as nx
#import metis
from collections import defaultdict
from utils import load_graph_file
import nxmetis


# def get_partition(file_name, vol=False):
#     """ Draws a partition using metis for python library"""
#
#     adj_list = load_graph_file(file_name)
#     G = nx.from_dict_of_lists(adj_list)
#     (cutcost, parts) = metis.part_graph(G, 3, objtype="cut") if vol == False else metis.part_graph(G, 3, objtype="vol")
    
#     if vol == False:
#         nx.nx_pydot.write_dot(G, "graph_files/" + file_name + "-cut.dot") # Requires pydot or pygraphviz
#         print(
#             f"The Graph {file_name} has been partitioned with cut={cutcost}"
#             )
#     else:
#         nx.nx_pydot.write_dot(G, "graph_files/" + file_name + "-vol.dot") # Requires pydot or pygraphviz
#         print(
#             f"The Graph {file_name} has been partitioned according to vol with cut={cutcost}"
#             )
    
#     return G, cutcost, parts


# def draw_partition(G, parts, output_file_name):
#     """ Draws a partition using metis for python library"""
#
#     colors = ['red','blue','green']
#     for i, p in enumerate(parts):
#         G.nodes[i+1]['color'] = colors[p]
#     print("Drawing the partitioned graph")
#     A = nx.nx_agraph.to_agraph(G)
#     A.node_attr['style']='filled'
#     #A.layout()
#     A.draw("images/" + output_file_name + ".png", prog="sfdp")
#     print(
#         f"Image succesfully saved to {'images/' + output_file_name + '.png'}"
#         )


# def compute_partitions(G, parts):
#     partitions = defaultdict(set)
#     for i, node in enumerate(G.nodes):
#         partitions[parts[i]].add(node)
#         #print(parts[i], f"Position {i} in nodes", node, f"Node label")
#     #print(parts)
#     for key in partitions.keys():
#         print(f"Lenght of partition #{key}:{len(partitions[key])}")


def get_partition(file_name, vol=False):
    """Partitions a graph using networkx-metis lib"""

    adj_list = load_graph_file(file_name)
    G = nx.from_dict_of_lists(adj_list)

    if vol == False:
        options = nxmetis.MetisOptions(ncuts=3, objtype=0)
        (cutcost, parts) = nxmetis.partition(G, nparts=3, options=options)
        nx.nx_pydot.write_dot(G, "graph_files/" + file_name + "-cut.dot") # Requires pydot or pygraphviz
        print(
            f"The Graph {file_name} has been partitioned with cut={cutcost}"
            )
    else:
        options = nxmetis.MetisOptions(ncuts=3, objtype=1)
        (cutcost, parts) = nxmetis.partition(G, nparts=3, options=options)
        nx.nx_pydot.write_dot(G, "graph_files/" + file_name + "-vol.dot") # Requires pydot or pygraphviz
        print(
            f"The Graph {file_name} has been partitioned according to vol with cut={cutcost}"
            )

    for i,p in enumerate(parts):
        print(f"Lenght of partition #{i}:{len(p)}")

    return G, cutcost, parts



def draw_partition(G, parts, output_file_name):
    """Draw a partition using networkx-metis lib"""

    colors = ['red','blue','green']
    for i, p in enumerate(parts):
        for j in p:
            G.nodes[j]['color'] = colors[i]

    print("Drawing the partitioned graph")
    A = nx.nx_agraph.to_agraph(G)
    A.node_attr['style']='filled'
    #A.layout()
    A.draw("images/" + output_file_name + ".png", prog="sfdp")
    print(
        f"Image succesfully saved to {'images/' + output_file_name + '.png'}"
        )
    

def graph_partitioning(file_name, vol=False):
    """Run Metis algorithm to create a 3-partition of a graph and draws it"""
    G, cutcost, parts = get_partition(file_name, vol)
    #compute_partitions(G, parts)
    if vol == False:
        draw_partition(G, parts, file_name + "-part-cut")
    else:
        draw_partition(G, parts, file_name + "-part-vol")
    print(
        "------------------------------ Process finished ------------------------------"
        )


if __name__ == "__main__":
    #file_name = "small_graphs/crack"
    #file_name = "t60k"
    file_name = "598a"
    graph_partitioning(file_name, False)
    graph_partitioning(file_name, True)
    
