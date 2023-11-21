# coding=gbk
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

NODE_NUMBER = 7115
MAX_PATH = 10000
r_sources = [10, 200, 40, 600, 8, 456, 855]
RUMOR_NUMBER = 7
CRUMOR_NUMBER = 7


def get_graph(G, node_number, r_sources, c_sources):
    # G = nx.DiGraph()
    # H = nx.path_graph(node_number)
    # G.add_nodes_from(H)

    # 初始化节点，并初始化状态
    for i in range(node_number):
        if i in r_sources:
            G.add_node(i, status='red', time=MAX_PATH)
        elif i in c_sources:
            G.add_node(i, status='green', time=MAX_PATH)
        else:
            G.add_node(i,status='yellow',time=MAX_PATH)
    # 添加边，并初始化权值
    # for i in range(node_number):
    #     for j in range(node_number):
    #         if i != j:
    #             rand_edge(i, j)
    return G


data = pd.read_csv('../data/wiki_vote/wiki_vote_matrix.csv')
matrix = np.asarray(data)
# 生成社交网络图
G = nx.from_numpy_matrix(matrix)
G = get_graph(G, NODE_NUMBER, r_sources, [])

logging.basicConfig(level=logging.INFO,
                        filename='../log/late_start_log.txt',
                        filemode='w',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def find_min_index(temple_length, node):
    min_index = -1
    min_length = MAX_PATH
    for i in range(RUMOR_NUMBER):
        if temple_length[i][node] < min_length:
            min_index = i
            min_length = temple_length[i][node]
    if min_index == -1:
        min_index = 0
    return min_index

'''
求谣言种子节点到所有节点的单源最短路径
并将路径也存下来
'''
temple_length = []
temple_path = []
for i in r_sources:
    one_of_shortest_path_length = []
    one_of_shortest_path = []
    for j in range(NODE_NUMBER):
        try:
            length = nx.dijkstra_path_length(G, i, j)

            path = nx.dijkstra_path(G, i, j)

            one_of_shortest_path_length.append(length)
            one_of_shortest_path.append(path)
            logging.info("path:{}, length:{}".format(path, length))
        except:
            one_of_shortest_path_length.append(MAX_PATH)
            one_of_shortest_path.append([-1])
            logging.info("path:{}, length:{}".format([-1], MAX_PATH))
    temple_length.append(one_of_shortest_path_length)
    temple_path.append(one_of_shortest_path)

print(temple_length)
print(temple_path)

length = []
path = []

'''找出最短路径长度和它经过的节点'''
for i in range(NODE_NUMBER):
    index = find_min_index(temple_length, i)
    length.append(temple_length[index][i])
    path.append(temple_path[index][i])

print(length)
print(len(length))
print(path)
print(len(path))

for i in range(NODE_NUMBER):
    print("path:{},length:{}".format(path[i], length[i]))


'''length就是感染时间列表'''
'''用一个字典来记录每个时间段有哪些节点感染'''

max_time = int(max(length))
print(max(length))

infect_time_dict = {}
for i in range(max_time+1):
    infect_time_dict[i] = []

print(infect_time_dict)

for i in range(len(length)):
    infect_time_dict[length[i]].append(i)

print(infect_time_dict)
for i in range(len(infect_time_dict)):
    # print("{}:{}".format(i, infect_time_dict[i]))
    logging.info("{}:{}".format(i, infect_time_dict[i]))