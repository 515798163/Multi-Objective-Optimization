import numpy as np
import pandas as pd
import dgl
import torch as th
import csv
import random
import warnings
import scipy.sparse as sp


def fast_calculate_new(feature_matrix, neighbor_num):
    """
    :param feature_matrix:
    :param neighbor_num: neighbor_num: must be less or equal than n-1 !!!!(n is the row count of feature matrix
    :return:
    """
    iteration_max = 50
    mu = 6
    X = feature_matrix
    alpha = np.power(X, 2).sum(axis=1)
    temp = alpha + alpha.T - 2 * X * X.T
    temp[np.where(temp < 0)] = 0
    distance_matrix = np.sqrt(temp)
    row_num = X.shape[0]
    e = np.ones((row_num, 1))
    distance_matrix = np.array(distance_matrix + np.diag(np.diag(e * e.T * np.inf)))
    sort_index = np.argsort(distance_matrix, kind='mergesort')
    nearest_neighbor_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num, row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(neighbor_num), nearest_neighbor_index] = 1
    C = nearest_neighbor_matrix
    np.random.seed(1234)
    W = np.mat(np.random.rand(row_num, row_num), dtype=float)
    W = np.multiply(C, W)
    lamda = mu * e
    P = X * X.T + lamda * e.T
    for q in range(iteration_max):
        Q = W * P
        W = np.multiply(W, P) / Q
        W = np.nan_to_num(W)
    return W


def calculate_linear_neighbor_simi(feature_matrix, neighbor_rate):
    """
    :param feature_matrix:
    :param neighbor_rate:
    :return:
    """
    neighbor_num = int(neighbor_rate * feature_matrix.shape[0])
    return fast_calculate_new(feature_matrix, neighbor_num)

def normalize_by_divide_rowsum(simi_matrix):
    simi_matrix_copy = np.matrix(simi_matrix, copy=True)
    for i in range(simi_matrix_copy.shape[0]):
        simi_matrix_copy[i, i] = 0
    row_sum_matrix = np.sum(simi_matrix_copy, axis=1)
    result = np.divide(simi_matrix_copy, row_sum_matrix)
    result[np.where(row_sum_matrix == 0)[0], :] = 0
    return result
def complete_linear_neighbor_simi_matrix(train_association_matrix, neighbor_rate):
    b = np.matrix(train_association_matrix)
    final_simi = calculate_linear_neighbor_simi(b, neighbor_rate)
    normalized_final_simi = normalize_by_divide_rowsum(
        final_simi)
    return normalized_final_simi
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data():
    D_SSM1 = np.loadtxt('data' + '/D_SSM1.txt')
    D_SSM2 = np.loadtxt('data' + '/D_SSM2.txt')
    D_GSM = np.loadtxt('data'+ '/D_GSM.txt')
    M_FSM = np.loadtxt('data' + '/M_FSM.txt')
    M_GSM = np.loadtxt('data' + '/M_GSM.txt')
    D_SSM = (D_SSM1 + D_SSM2) / 2

    ID = np.zeros(shape=(D_SSM.shape[0], D_SSM.shape[1]))  ##疾病的相似性矩阵
    IM = np.zeros(shape=(M_FSM.shape[0], M_FSM.shape[1]))   ##mirna的相似性矩阵
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
            if D_SSM[i][j]== 0:
                ID[i][j] = D_GSM[i][j]
            else:
                ID[i][j] = D_SSM[i][j]
    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if M_FSM[i][j] == 0:
                IM[i][j] = M_GSM[i][j]
            else:
                IM[i][j] = M_FSM[i][j]

    return ID, IM
    csv_path = 'C:\\Users\\main\\Desktop\\APDGR\\data\\disease name.csv'
    with open(csv_path, 'r', encoding='utf8') as fp:
        dis_idx = [i[1] for i in csv.reader(fp)]
    dis_idx == np.array(dis_idx)
    dis_map_idx = {j: i for i, j in enumerate(dis_idx,1)}

    csv1_path = 'C:\\Users\\main\\Desktop\\APDGR\\data\\miRNA number.csv'
    with open(csv1_path, 'r', encoding='utf8') as fp:
        mrna_idx = [j[1] for j in csv.reader(fp)]
    mrna_idx = np.array(mrna_idx)
    mrna_map_idx = {j: i for i, j in enumerate(mrna_idx,1)}

    csv2_path = 'C:\\Users\\main\\Desktop\\APDGR\\data\\miRNA-disease association.csv'  #####读联合数据
    with open(csv2_path, 'r', encoding='utf8') as fp:
        mrna_dis_idx = [j for j in csv.reader(fp)]

    adj_mrna_dis = []
    for i in mrna_dis_idx:  ####将mrna——disease的联合转为列表，并以编号节点的形式表示
        adj_mrna_dis.append([mrna_map_idx[i[0]], dis_map_idx[i[1]]])

    """构建一个疾病——RNA矩阵，疾病有384种，RNA有495种，对应的相互作用有5430，形成一个495*384的矩阵"""
    adj_mrna_dis = np.array(adj_mrna_dis)  ###将疾病——mrna 的联合转换成一个矩阵
    adj_mrna_dis = sp.coo_matrix((np.ones(adj_mrna_dis.shape[0]), (adj_mrna_dis[:, 0], adj_mrna_dis[:, 1])),
                                 shape=(496, 384), dtype=np.float32)  ###从0行，0列开始，所以行和列维度都+1

    csv3_path = 'C:\\Users\\main\\Desktop\\APDGR\\data\\gens.csv'
    with open(csv3_path, 'r', encoding='utf8') as fp:
        gens_idx = np.array([i[1] for i in csv.reader(fp)])
    gens_map_idx = {j: i for i, j in enumerate(gens_idx,1)}

    adj_gens_dis = []
    csv4_path = 'C:\\Users\\main\\Desktop\\APDGR\\data\\diseaseid-disease-gene.csv'  #####读gen_disease联合数据
    with open(csv4_path, 'r', encoding='utf8') as fp:
        gens_dis_idx = [j for j in csv.reader(fp)]
    for i in gens_dis_idx:
        adj_gens_dis.append([gens_map_idx[i[2]], dis_map_idx[i[1]]])

    adj_gens_dis = np.array(adj_gens_dis)

    adj_gens_dis = np.array(adj_gens_dis)  ###构建 gens——disease 联合矩阵
    """构建一个基因——疾病矩阵，基因有4519个，疾病有383种，对应的相互作用有18818，形成一个384*4520的矩阵"""
    adj_dis_gens = sp.coo_matrix((np.ones(adj_gens_dis.shape[0]), (adj_gens_dis[:, 1], adj_gens_dis[:, 0])),
                                 shape=(384, 4520), dtype=np.float32)
    ID1=adj_dis_gens.todense()
    q=ID1[1:,1:]
    ID1=complete_linear_neighbor_simi_matrix(q, 0.9)
    ID1=normalize(ID1)
    IM1=adj_mrna_dis.todense()
    IM1=IM1[1:,1:]
    IM1=complete_linear_neighbor_simi_matrix(IM1, 0.9)
    IM1=normalize(IM1)
    ID=(ID+ID1)/2
    IM = (IM + IM1) / 2
    return ID, IM


def sample(directory, random_seed):
    all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])

    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)
    return sample_df.values


def build_graph(directory, random_seed):

    # ID, IM = load_data(directory)
    ID, IM = load_data()
    samples = sample(directory, random_seed)

    print('Building graph ...')
    warnings.filterwarnings('ignore')
    g = dgl.DGLGraph()
    g.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = th.zeros(g.number_of_nodes(), dtype=th.float32)
    node_type[:ID.shape[0]] = 1
    g.ndata['type'] = node_type

    print('Adding disease features ...')
    d_data = th.zeros((g.number_of_nodes(), ID.shape[1]),dtype=th.float32)
    d_data[: ID.shape[0], :] = th.from_numpy(ID)
    g.ndata['d_features'] = d_data

    print('Adding miRNA features ...')
    m_data = th.zeros((g.number_of_nodes(), IM.shape[1]), dtype=th.float32)
    m_data[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = th.from_numpy(IM)
    g.ndata['m_features'] = m_data

    print('Adding edges ...')
    disease_ids = list(range(1, ID.shape[0] + 1))
    mirna_ids = list(range(1, IM.shape[0] + 1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]    #####sample_disease_vertices代表疾病编码后的索引0--382
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]   #####sample_mirna_vertices代表RNA结点编码后的索引 383--383+495

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,  ##添加边，，构造无向图
                    data={'inv': th.zeros(samples.shape[0], dtype=th.int32),
                          'rating': th.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                    data={'inv': th.zeros(samples.shape[0], dtype=th.int32),
                          'rating': th.from_numpy(samples[:, 2].astype('float32'))})

    g.readonly()
    print('Successfully build graph !!')
    return g, disease_ids_invmap, mirna_ids_invmap

