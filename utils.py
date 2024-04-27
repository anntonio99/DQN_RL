import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import networkx as nx
import random





def create_simple_graph():
  G = nx.Graph()
  G.add_nodes_from([0, 1, 2, 3])
  G.add_edges_from([(0,1), (1,2), (2,3), (3,1)])

  for i, j in G.edges():
        G.get_edge_data(i, j)["capacity"] = float(200)
        G.get_edge_data(i, j)['bandwidth_allocated'] = 0
  return G

def create_simple_graph_2():
  G = nx.Graph()
  G.add_nodes_from([0, 1, 2, 3, 4])
  G.add_edges_from([(0,1), (1,2), (2,3), (3,1), (2,4), (3,4)])

  for i, j in G.edges():
        G.get_edge_data(i, j)["capacity"] = float(200)
        G.get_edge_data(i, j)['bandwidth_allocated'] = 0
  return G

def create_complex_graph():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    G.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12,19), (12,21),
         (14, 15), (15, 16), (16, 17), (17,18), (18,21), (19, 23), (21,22), (22, 23)])
    for i, j in G.edges():
        G.get_edge_data(i, j)["capacity"] = float(200)
        G.get_edge_data(i, j)['bandwidth_allocated'] = 0
    return G









def generate_random_graph(n):
    # Generate a random connected graph
    G = nx.connected_watts_strogatz_graph(n, random.randint(1, n // 2), 0.2)

    # Ensure the graph is connected
    while not nx.is_connected(G):
        # Randomly select two nodes
        nodes = list(G.nodes)
        node1, node2 = random.sample(nodes, 2)
        # Add an edge between them if it doesn't exist
        if not G.has_edge(node1, node2):
            G.add_edge(node1, node2)
    
    for i, j in G.edges():
        G.get_edge_data(i, j)["capacity"] = float(200)
        G.get_edge_data(i, j)['bandwidth_allocated'] = 0

    return G