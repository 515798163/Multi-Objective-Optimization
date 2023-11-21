import random
import networkx as nx
import pandas as pd
import numpy as np

data = pd.read_csv('matrix.csv')
matrix = np.asarray(data)

G = nx.from_numpy_matrix(matrix)

print(G.nodes())

MAX_PATH = 10000
shortest_path_lists = []
for i in G.nodes():
    one_of_shortest_path = []
    for j in G.nodes():
        try:
            one_of_shortest_path.append(nx.dijkstra_path_length(G, i, j))
            print("{}->{}".format(i, j))
        except:
            one_of_shortest_path.append(MAX_PATH)
            print("{}->{}".format(i, j))
    shortest_path_lists.append(one_of_shortest_path)
shortest_path = np.array(shortest_path_lists)
print(shortest_path)

pd.DataFrame(shortest_path).to_csv('shortest_path.csv', header=True, index=False)
print(shortest_path.shape)

