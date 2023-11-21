import random
import networkx as nx
import pandas as pd
import numpy as np
import logging

MAX_PATH = 10000

data = pd.read_csv('facebook_shortest_path.csv')
matrix = np.asarray(data)
print(matrix.dtype)

print(matrix.shape)

for i in range(4039):
    for j in range(4039):
        if matrix[i, j] == np.inf:
            matrix[i, j] = MAX_PATH

matrix = matrix.astype(np.int64)

print(matrix)
print(matrix.dtype)

pd.DataFrame(matrix).to_csv('facebook_path.csv', header=True, index=False)
print(matrix.shape)