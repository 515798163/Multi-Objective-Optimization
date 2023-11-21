import numpy as np

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
    np.random.seed(0)
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

D_SSM1 = np.loadtxt('data' + '/D_SSM1.txt')
D_SSM2 = np.loadtxt('data' + '/D_SSM2.txt')
D_GSM = np.loadtxt('data' + '/D_GSM.txt')
M_FSM = np.loadtxt('data' + '/M_FSM.txt')
M_GSM = np.loadtxt('data' + '/M_GSM.txt')
D_SSM = (D_SSM1 + D_SSM2) / 2

ID = np.zeros(shape=(D_SSM.shape[0], D_SSM.shape[1]))  ##疾病的相似性矩阵
IM = np.zeros(shape=(M_FSM.shape[0], M_FSM.shape[1]))  ##mirna的相似性矩阵
for i in range(D_SSM.shape[0]):
    for j in range(D_SSM.shape[1]):
        if D_SSM[i][j] == 0:
            ID[i][j] = D_GSM[i][j]
        else:
            ID[i][j] = D_SSM[i][j]
for i in range(M_FSM.shape[0]):
    for j in range(M_FSM.shape[1]):
        if M_FSM[i][j] == 0:
            IM[i][j] = M_GSM[i][j]
        else:
            IM[i][j] = M_FSM[i][j]

alpha = 0.1
neighbor_rate = 0.9
weight = 1.0




W=calculate_linear_neighbor_simi(ID, neighbor_rate)
a=complete_linear_neighbor_simi_matrix(ID, neighbor_rate)
print('1111')
print('1111')