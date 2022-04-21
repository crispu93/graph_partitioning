from collections import defaultdict


def load_graph_file(file_name):
    path = "graph_files/" + file_name + ".graph"
    adj_list = defaultdict(list)
    with open(path) as fp:
        fp.readline()
        for i, line in enumerate(fp):
            adj_nodes = [int(x) for x in line.strip().split()]
            adj_list[i+1].extend(adj_nodes)
            #print(i+1, ":", adj_list[i+1])
    return adj_list

if __name__ == "__main__":
    load_graph_file("add20")