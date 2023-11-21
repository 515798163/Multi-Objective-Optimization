import torch as th
import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from layers import GraphSageLayer
from dgl.nn import SAGEConv
from dgl.nn import EdgePredictor


class GNNMDA(nn.Module):
    def __init__(self, encoder, decoder):
        super(GNNMDA, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, diseases, mirnas):
        h = self.encoder(G)
        h_diseases = h[diseases]
        h_mirnas = h[mirnas]

        return self.decoder(h_diseases, h_mirnas)


#####调用dgl库中的GraphSAGE来进行聚合

class GraphSEncoder(nn.Module):
    def __init__(self,G, embedding_size,dropout):
        super(GraphSEncoder, self).__init__()
        self.G=G
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.disease_emb = DiseaseEmbedding(embedding_size, dropout)
        self.mirna_emb = MirnaEmbedding(embedding_size, dropout)

        # G.apply_nodes(lambda nodes: {'h': self.disease_emb(nodes.data)}, self.disease_nodes)
        # G.apply_nodes(lambda nodes: {'h': self.mirna_emb(nodes.data)}, self.mirna_nodes)

        self.conv1 = SAGEConv(embedding_size,128, 'mean') ###'mean'
        self.conv2 = SAGEConv(128, 128, 'mean')  ###'mean'
    def forward(self, G):
        G.apply_nodes(lambda nodes: {'h': self.disease_emb(nodes.data)}, self.disease_nodes)
        G.apply_nodes(lambda nodes: {'h': self.mirna_emb(nodes.data)}, self.mirna_nodes)

        h = self.conv1(G, G.ndata['h'])
        h = F.relu(h)
        h = self.conv2(G, h)
        return h

class DiseaseEmbedding(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(DiseaseEmbedding, self).__init__()

        seq = nn.Sequential(
            nn.Linear(383, embedding_size),
            # nn.Linear(4519, embedding_size),
            nn.Dropout(dropout)
        )
        self.proj_disease = seq

    def forward(self, ndata):
        with torch.no_grad():
            extra_repr = self.proj_disease(ndata['d_features'])
        return extra_repr

class MirnaEmbedding(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(MirnaEmbedding, self).__init__()

        seq = nn.Sequential(
            nn.Linear(495, embedding_size),
            # nn.Linear(383, embedding_size),
            nn.Dropout(dropout)
        )
        self.proj_mirna = seq

    def forward(self, ndata):
        with torch.no_grad():
            extra_repr = self.proj_mirna(ndata['m_features'])
        return extra_repr

class BilinearDecoder(nn.Module):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        self.activation=nn.Sigmoid()
        # self.W = nn.Parameter(torch.FloatTensor(feature_size, feature_size))
        self.W = nn.Parameter(torch.FloatTensor(128,128))

        self.init_params()
    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self,h_diseases, h_mirnas):
        # predictor =EdgePredictor('cos')  ### ‘cos’--预测AUC 0.9349 单独用这个
        # result1=predictor(h_diseases, h_mirnas).sum(1)
        # results_mask=predictor(h_diseases, h_mirnas).sum(1)

        results_mask = self.activation((th.mm(h_diseases, self.W) * h_mirnas).sum(1))  ##代表按照行相加起来 0.9363
        # results_mask=(results_mask+result1)/2

        return results_mask
