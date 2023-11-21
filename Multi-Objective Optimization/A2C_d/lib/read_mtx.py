# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np

import scanpy as sc
adata = sc.read('../data/Amherst41/socfb-Amherst41.mtx')
data = adata.X
print(data)
data = np.asarray(data.todense())
print(data)

# 读取mtx文件
G = nx.from_numpy_matrix(data)

# 输出节点数和边数
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())



