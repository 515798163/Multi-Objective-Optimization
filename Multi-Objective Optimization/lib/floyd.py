import random
import networkx as nx
import pandas as pd
import numpy as np
import logging


# logging.basicConfig(level=logging.INFO,
#                     filename='./face_log.txt',
#                     filemode='w',
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

data = pd.read_csv('facebook_matrix.csv')
matrix = np.asarray(data)

G = nx.from_numpy_matrix(matrix)

print(G.nodes())

shortest_path = nx.floyd_warshall_numpy(G)

pd.DataFrame(shortest_path).to_csv('facebook_shortest_path.csv', header=True, index=False)
print(shortest_path.shape)

