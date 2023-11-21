# coding: utf-8
import networkx as nx
import scipy.io

# 读取.mtx文件
matrix = nx.read_mtx('../data/Amherst41/socfb-Amherst41.mtx')

graph = nx.from_scipy_sparse_matrix(matrix)