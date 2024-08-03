import networkx as nx
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx

print(torch.__version__)

# reading the graph from text
G = nx.read_edgelist("facebook_combined.txt")


# visualization of graph
#nx.draw(G)
#plt.show()

# convert into torch tensor
pyg_graph = from_networkx(G)

print(pyg_graph.edge_index.shape)