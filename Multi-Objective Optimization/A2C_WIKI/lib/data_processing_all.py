#-*-coding:gb2312-*-
import numpy as np
import pandas as pd
import networkx as nx
import logging
import random
import scipy.sparse as sp

PROBABILITY = 2
MAX_PATH = 10000
NODE_NUMBER = 4039
r_sources = [10, 200, 40, 600, 8]
# r_sources = [10, 200]
RUMOR_NUMBER = len(r_sources)

SOURCE_TARGET_PATH = '../data/facebook_combined.csv'
MATRIX_PATH = '../data/facebook_matrix.csv'
SHORTEST_PATH = '../data/facebook_shortest_path.csv'

DEGREE_PATH = '../data/facebook_degree_centrality.csv'
EIGENVECTOR_PATH = '../data/facebook_eigenvector_centrality.csv'
CLOSENESS_PATH = '../data/facebook_closeness_centrality.csv'
PAGERANK_PATH = '../data/facebook_pagerank.csv'
HITS_PATH = '../data/facebook_hits.csv'
CLUSTERING_PATH = '../data/facebook_clustering.csv'
BETWEENNESS_PATH = '../data/facebook_betweenness_centrality.csv'
CONTRIBUTE_PATH = '../data/facebook_contribute.csv'

FEATURES_PATH = '../data/facebook_features.csv'

NORMALIZE_FEATURE_PATH = '../data/facebook_nprmalize_features.csv'

NORMALIZE_all_FEATURE_PATH = '../data/facebook_nprmalize_all_features.csv'



'''生成邻接矩阵'''
def Generate_an_adjacency_matrix():
    logging.basicConfig(level=logging.INFO,
                        filename='../log/Sloshdot.txt',
                        filemode='w',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    data = pd.read_csv(SOURCE_TARGET_PATH)

    adjacency_list = np.asarray(data)

    print(adjacency_list)

    matrix = np.zeros((NODE_NUMBER, NODE_NUMBER))
    print(matrix.shape)

    for edge in adjacency_list:
        x = random.randint(1, 10)
        if x <= PROBABILITY:
            weight = random.randint(1, 10)
        else:
            weight = MAX_PATH

        matrix[edge[0], edge[1]] = weight
        print('{}->{}={}'.format(edge[0], edge[1], weight))
        logging.info('{}->{}={}'.format(edge[0], edge[1], weight))

    print(matrix)

    pd.DataFrame(matrix).to_csv(MATRIX_PATH, header=True, index=False)

'''处理最短路径'''
def get_shortest_path():
    data = pd.read_csv(MATRIX_PATH)
    matrix = np.asarray(data)

    G = nx.from_numpy_matrix(matrix)

    print(G.nodes())

    shortest_path = nx.floyd_warshall_numpy(G)

    pd.DataFrame(shortest_path).to_csv(SHORTEST_PATH, header=True, index=False)
    print(shortest_path.shape)

'''处理特征'''
def get_feature():
    degree_centrality_np = np.zeros(4039)
    eigenvector_centrality_np = np.zeros(4039)
    closeness_centrality_np =  np.zeros(4039)
    pagerank_np =  np.zeros(4039)
    hits_np = np.zeros(4039)
    clustering_np = np.zeros(4039)
    betweenness_centrality_np = np.zeros(4039)





    data = pd.read_csv(MATRIX_PATH)
    matrix = np.asarray(data)

    G = nx.from_numpy_matrix(matrix)

    print(G.nodes())

    degree_centrality = nx.degree_centrality(G) #度中心性
    print(degree_centrality)
    for i in range(NODE_NUMBER):
        degree_centrality_np[i] = degree_centrality[i]
    pd.DataFrame(degree_centrality_np).to_csv(DEGREE_PATH, header=True, index=False)


    eigenvector_centrality = nx.eigenvector_centrality(G) #特征向量中心性
    print(eigenvector_centrality)
    for i in range(NODE_NUMBER):
        eigenvector_centrality_np[i] = eigenvector_centrality[i]
    pd.DataFrame(eigenvector_centrality_np).to_csv(EIGENVECTOR_PATH, header=True, index=False)

    closeness_centrality = nx.closeness_centrality(G)   #连接中心性
    print(closeness_centrality)
    for i in range(NODE_NUMBER):
        closeness_centrality_np[i] = closeness_centrality[i]
    pd.DataFrame(closeness_centrality_np).to_csv(CLOSENESS_PATH, header=True, index=False)


    pagerank = nx.pagerank(G)   #pagerank
    print(pagerank)
    for i in range(NODE_NUMBER):
        pagerank_np[i] = pagerank[i]
    pd.DataFrame(pagerank_np).to_csv(PAGERANK_PATH, header=True, index=False)

    # hits = nx.hits(G)   #hits
    # print(hits)
    # for i in range(NODE_NUMBER):
    #     hits_np[i] = hits[i]
    # pd.DataFrame(hits_np).to_csv(HITS_PATH, header=True, index=False)

    clustering = nx.clustering(G)   #节点的聚类系数
    print(clustering)
    for i in range(NODE_NUMBER):
        clustering_np[i] = clustering[i]
    pd.DataFrame(clustering_np).to_csv(CLUSTERING_PATH, header=True, index=False)

    betweenness_centrality = nx.betweenness_centrality(G)   #中介中心性
    print(betweenness_centrality)
    for i in range(NODE_NUMBER):
        betweenness_centrality_np[i] = betweenness_centrality[i]
    pd.DataFrame(betweenness_centrality_np).to_csv(BETWEENNESS_PATH, header=True, index=False)

'''计算贡献度'''
def get_contribute():
    data = pd.read_csv(MATRIX_PATH)
    matrix = np.asarray(data)
    G = nx.from_numpy_matrix(matrix)
    G = get_graph(G, NODE_NUMBER, r_sources, [])

    logging.basicConfig(level=logging.INFO,
                        filename='../log/contribute.txt',
                        filemode='w',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    '''
    求谣言种子节点到所有节点的单源最短路径
    并将路径也存下来
    '''
    temple_length = []
    temple_path = []

    '''获取最短路径'''
    shortest_path = pd.read_csv(SHORTEST_PATH)
    shortest_path_np = np.asarray(shortest_path)
    path_list = shortest_path_np.tolist()

    for source in r_sources:
        temple_length.append(path_list[source])
    print(temple_path)
    logging.info("shortest_path:{}".format(temple_length))

    '''记录路径'''
    for i in r_sources:
        one_of_shortest_path = []
        for j in range(NODE_NUMBER):
            try:
                path = nx.dijkstra_path(G, i, j)
                one_of_shortest_path.append(path)
                logging.info("path:{}".format(path))
            except:
                one_of_shortest_path.append([-1])
                logging.info("path:{}".format([-1]))
        temple_path.append(one_of_shortest_path)
        print(i)
    print(temple_path)

    length = []
    path = []

    '''找出最短路径长度和它经过的节点'''
    for i in range(NODE_NUMBER):
        index = find_min_index(temple_length, i)
        if index == -1:
            length.append(MAX_PATH)
            path.append([])
        else:
            length.append(temple_length[index][i])
            path.append(temple_path[index][i])

    print(length)
    print(len(length))
    print(path)
    print(len(path))

    '''记录每个节点激活了哪些节点'''
    contribute_table = np.zeros((NODE_NUMBER, NODE_NUMBER))

    for i in range(NODE_NUMBER):
        for one_of_path in path:
            list_length = len(one_of_path)
            for j in range(list_length):
                if one_of_path[j] == i and j < list_length - 1:
                    contribute_table[i][one_of_path[j + 1]] += 1

    print(contribute_table)

    '''找出每个节点的贡献度'''
    contribute = np.zeros(NODE_NUMBER)
    for i in range(NODE_NUMBER):
        for j in range(NODE_NUMBER):
            if contribute_table[i][j] != 0:
                contribute[i] += 1
    print(contribute)

    pd.DataFrame(contribute).to_csv(CONTRIBUTE_PATH, header=True, index=False)

'''数据归一化'''
def normalize(mx):
    # mx = np.asarray(mx)
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #  矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx

'''组装特征'''
def assemble_feature():
    feature_list = []

    '''组装degree_centrality度中心性'''
    degree_centrality = pd.read_csv(DEGREE_PATH)
    degree_centrality_np = np.asarray(degree_centrality)
    degree_centrality_np = np.squeeze(degree_centrality_np)
    feature_list.append(degree_centrality_np)

    '''组装eigenvector_centrality特征向量中心性'''
    eigenvector_centrality = pd.read_csv(EIGENVECTOR_PATH)
    eigenvector_centrality_np = np.asarray(eigenvector_centrality)
    eigenvector_centrality_np = np.squeeze(eigenvector_centrality_np)
    feature_list.append(eigenvector_centrality_np)

    '''组装closeness_centrality'''
    closeness_centrality = pd.read_csv(CLOSENESS_PATH)
    closeness_centrality_np = np.asarray(closeness_centrality)
    closeness_centrality_np = np.squeeze(closeness_centrality_np)
    feature_list.append(closeness_centrality_np)

    '''组装pagerank'''
    pagerank = pd.read_csv(PAGERANK_PATH)
    pagerank_np = np.asarray(pagerank)
    pagerank_np = np.squeeze(pagerank_np)
    feature_list.append(pagerank_np)

    '''组装clustering'''
    clustering = pd.read_csv(CLUSTERING_PATH)
    clustering_np = np.asarray(clustering)
    clustering_np = np.squeeze(clustering_np)
    feature_list.append(clustering_np)

    '''组装betweenness_centrality'''
    betweenness_centrality = pd.read_csv(BETWEENNESS_PATH)
    betweenness_centrality_np = np.asarray(betweenness_centrality)
    betweenness_centrality_np = np.squeeze(betweenness_centrality_np)
    feature_list.append(betweenness_centrality_np)

    '''组装contribute'''
    contribute = pd.read_csv(CONTRIBUTE_PATH)
    contribute_np = np.asarray(contribute)
    contribute_np = np.squeeze(contribute_np)
    feature_list.append(contribute_np)

    # feature_list = normalize(feature_list)

    '''组装谣言标签'''
    rumor_np = np.zeros(NODE_NUMBER)
    for i in r_sources:
        rumor_np[i] = 1
    feature_list.append(rumor_np)

    '''组装反谣言标签'''
    crumor_np = np.zeros(NODE_NUMBER)
    feature_list.append(crumor_np)

    # for element in feature_list:
    #     print(element.shape)
    #     print(type(element))

    features = np.asarray(feature_list)
    print(features.shape)
    features = features.T
    print(features.shape)
    features_normalize = normalize(features)
    print(features_normalize.shape)


    pd.DataFrame(features_normalize).to_csv(NORMALIZE_all_FEATURE_PATH, header=True, index=False)


def get_graph(G, node_number, r_sources, c_sources):
    # 初始化节点，并初始化状态
    for i in range(node_number):
        if i in r_sources:
            G.add_node(i, status='red', time=MAX_PATH)
        elif i in c_sources:
            G.add_node(i, status='green', time=MAX_PATH)
        else:
            G.add_node(i,status='yellow',time=MAX_PATH)

    return G

def find_min_index(temple_length, node):
    min_index = -1
    min_length = MAX_PATH
    for i in range(RUMOR_NUMBER):
        if temple_length[i][node] < min_length:
            min_index = i
            min_length = temple_length[i][node]
    # if min_index == -1:
    #     min_index = 0
    return min_index

# Generate_an_adjacency_matrix()
# get_shortest_path()
# get_feature()
# get_contribute()
assemble_feature()

# shortest_path = pd.read_csv(CONTRIBUTE_PATH)
# print(shortest_path.sum())