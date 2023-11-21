import time
import random
import numpy as np
import pandas as pd
import math
import dgl
from sklearn.model_selection import KFold
from sklearn import metrics

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from model import GNNMDA,GraphEncoder, BilinearDecoder
from model import GNNMDA,GraphSEncoder, BilinearDecoder

from utils import build_graph, sample, load_data
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda')
cpu_device = torch.device('cpu')

def Train(directory, epochs, aggregator, embedding_size, layers, dropout, slope, lr, wd, random_seed):
    
    random.seed(random_seed)
    np.random.seed(random_seed)

    # mx.random.seed(random_seed)

    g, disease_ids_invmap, mirna_ids_invmap = build_graph(directory, random_seed=random_seed)  ##返回图g的属性，疾病和miRNA的索引
    samples = sample(directory, random_seed=random_seed)
    g = g.to(device)
    # ID, IM = load_data(directory)
    ID, IM = load_data()
    print('## vertices:', g.number_of_nodes())  ########g.number_of_nodes()=878
    print('## edges:', g.number_of_edges())         ########g.number_of_edges()=21720
    print('## disease nodes:', th.sum(g.ndata['type'] == 1))  ####383
    print('## mirna nodes:', th.sum(g.ndata['type'] == 0))   ####495

    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
    sample_disease_vertices = th.tensor([disease_ids_invmap[id_] for id_ in samples[:, 1]]).to(device)
    sample_mirna_vertices = th.tensor([mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]).to(device)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)   ###5倍交叉，在数据集中随机以1:4划分训练集和验证集
    train_index = []
    test_index = []
    for train_idx, test_idx in kf.split(samples[:, 2]):
        train_index.append(train_idx)
        test_index.append(test_idx)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []

    fprs = []
    tprs = []

    for i in range(len(train_index)):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)

        samples_df['train'] = 0   ####在数据集中加入train列，值置0
        samples_df['test'] = 0     ####在数据集中加入test列，值置0

        samples_df['train'].iloc[train_index[i]] = 1   ## iloc,指取出train_index的第i行 详解，https://www.jianshu.com/p/dadf2f1b88fc  train_index[0]代表次一次划分训练集时候在samples_df中的索引
        samples_df['test'].iloc[test_index[i]] = 1     ###取出对应于训练集的测试集
        #########上两句的作用就是在每一次的训练过程中划分得到训练集和测试集
        train_tensor = th.from_numpy(samples_df['train'].values.astype('int32')).to(device)
        test_tensor = th.from_numpy(samples_df['test'].values.astype('int32')).to(device)

        edge_data = {'train': train_tensor,
                     'test': test_tensor}

        g.edges[sample_disease_vertices, sample_mirna_vertices].data.update(edge_data)    ####识别出原图结构的训练集和测试集
        g.edges[sample_mirna_vertices, sample_disease_vertices].data.update(edge_data)

        train_eid = g.filter_edges(lambda edges: edges.data['train'])   ####取出训练集中对应边的ID返回给train_eid  #https://docs.dgl.ai/generated/dgl.DGLGraph.filter_edges.html
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True).to(device)    #https://docs.dgl.ai/generated/dgl.edge_subgraph.html#dgl.edge_subgraph,返回子图包括给定的边
        # g_train.copy_from_parent()   ##复制对应训练集的子图的属性

        # get the training set
        rating_train = g_train.edata['rating'].to(device)        ####   rating_train =训练集的标签
        # src_train, dst_train = g_train.all_edges()  ######返回训练子图中所有的边的起始节点和目标节点
        src_train, dst_train=g.find_edges(train_eid)
        src_train = src_train.to(device)
        dst_train = dst_train.to(device)
        # get the testing edge set
        test_eid = g.filter_edges(lambda edges: edges.data['test'])
        src_test, dst_test = g.find_edges(test_eid)
        src_test = src_test.to(device)
        dst_test = dst_test.to(device)
        rating_test = g.edges[test_eid].data['rating'].to(device)
        

        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        # Train the model
        # model = GNNMDA(GraphSEncoder(embedding_size=embedding_size, n_layers=layers, G=g_train, aggregator=aggregator,
        #                             dropout=dropout, slope=slope),
        #                BilinearDecoder(feature_size=embedding_size))
        model = GNNMDA(GraphSEncoder(embedding_size=embedding_size, G=g_train,
                                    dropout=dropout),
                       BilinearDecoder(feature_size=embedding_size)).to(device)

        # cross_entropy = nn.BCEWithLogitsLoss()##or
        # cross_entropy =nn.BCELoss()
        # cross_entropy =nn.CrossEntropyLoss()
        cross_entropy =nn.MSELoss()  ###用这个loss很低
        trainer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(epochs):
            start = time.time()
            for _ in range(10):

                trainer.zero_grad()
                score_train = model(g_train, src_train, dst_train).to(device)
                loss_train = cross_entropy(score_train, rating_train).mean()  ##or
                
                loss_train.backward()
                trainer.step()

            h_val = model.encoder(g).to(device)
            score_val = model.decoder(h_val[src_test], h_val[dst_test]).to(device)

            loss_val = cross_entropy(score_val, rating_test).mean()

            train_auc = metrics.roc_auc_score(np.squeeze(rating_train.to(cpu_device).detach().numpy()), np.squeeze(score_train.to(cpu_device).detach().numpy()))

            val_auc = metrics.roc_auc_score(np.squeeze(rating_test.to(cpu_device).detach().numpy()), np.squeeze(score_val.to(cpu_device).detach().numpy()))

            results_val = [0 if j < 0.4 else 1 for j in np.squeeze(score_val.to(cpu_device).detach().numpy())]

            accuracy_val = metrics.accuracy_score(rating_test.to(cpu_device).detach().numpy(), results_val)

            precision_val = metrics.precision_score(rating_test.to(cpu_device).detach().numpy(), results_val)

            recall_val = metrics.recall_score(rating_test.to(cpu_device).detach().numpy(), results_val)

            f1_val = metrics.f1_score(rating_test.to(cpu_device).detach().numpy(), results_val)

            end = time.time()

            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),   ####loss_train.asscalar()
                  'Val Loss: %.4f' % loss_val.item(),  ###loss_val.asscalar()
                  'Acc: %.4f' % accuracy_val, 'Pre: %.4f' % precision_val, 'Recall: %.4f' % recall_val,
                  'F1: %.4f' % f1_val, 'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc,
                  'Time: %.2f' % (end - start))

        model.eval()
        with torch.no_grad():
            h_test = model.encoder(g).to(device)
            score_test = model.decoder(h_test[src_test], h_test[dst_test]).to(device)


           

        # loss_test = cross_entropy(score_test, rating_test).mean()   ##zuozheyuanbend

        fpr, tpr, thresholds = metrics.roc_curve(np.squeeze(rating_test.to(cpu_device).detach().numpy()),
                                                 np.squeeze(score_test.to(cpu_device).detach().numpy()))

        test_auc = metrics.auc(fpr, tpr)

        results_test = [0 if j < 0.4 else 1 for j in np.squeeze(score_test.to(cpu_device).detach().numpy())]
        accuracy_test = metrics.accuracy_score(rating_test.to(cpu_device).detach().numpy(), results_test)
        precision_test = metrics.precision_score(rating_test.to(cpu_device).detach().numpy(), results_test)
        recall_test = metrics.recall_score(rating_test.to(cpu_device).detach().numpy(), results_test)
        f1_test = metrics.f1_score(rating_test.to(cpu_device).detach().numpy(), results_test)
  
        print('Fold:', i + 1,
              'Test Acc: %.4f' % accuracy_test, 'Test Pre: %.4f' % precision_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test AUC: %.4f' % test_auc
              )

        auc_result.append(test_auc)
        acc_result.append(accuracy_test)
        pre_result.append(precision_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)

        fprs.append(fpr)
        tprs.append(tpr)

    print('## Training Finished !')
    print(
        '----------------------------------------------------------------------------------------------------------')

    return auc_result, acc_result, pre_result, recall_result, f1_result, fprs, tprs
