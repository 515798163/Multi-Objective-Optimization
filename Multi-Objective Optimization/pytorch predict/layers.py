import dgl.function as FN
import numpy as np
import torch
import torch as th
import torch.nn as nn

class GraphSageLayer(nn.Module):
    def __init__(self, feature_size, G, disease_nodes, mirna_nodes, dropout, slope):
        super(GraphSageLayer, self).__init__()

        self.feature_size = feature_size
        self.G = G
        self.disease_nodes = disease_nodes
        self.mirna_nodes = mirna_nodes

        self.disease_update = NodeUpdate(feature_size, dropout, slope)
        self.miran_update = NodeUpdate(feature_size, dropout, slope)

        all_nodes = th.arange(G.number_of_nodes())
        self.deg = G.in_degrees(all_nodes)

    def forward(self, G):
        assert G.number_of_nodes() == self.G.number_of_nodes()
        G.ndata['deg'] = self.deg

        G.update_all(FN.copy_src('h', 'h'), FN.sum('h', 'h_agg'))  # mean, max, sum
            
        G.apply_nodes(self.disease_update, self.disease_nodes)
        G.apply_nodes(self.miran_update, self.mirna_nodes)


class NodeUpdate(nn.Module):
    def __init__(self, feature_size, dropout, slope):
        super(NodeUpdate, self).__init__()

        self.feature_size = feature_size
        self.leakyrelu = nn.LeakyReLU(slope)  ##m = nn.LeakyReLU(0.1)
        ###https: // pytorch - cn.readthedocs.io / zh / latest / package_references / torch - nn /  # non-linear-activations-source

        self.W = nn.Linear(feature_size*2, feature_size)
        self.dropout = nn.Dropout(dropout)  ###m = nn.Dropout(p=0.2)
        

    # def init_params(self):
    #     for param in self.parameters():
    #         nn.init.xavier_uniform_(param)

    def forward(self, nodes):
        h = nodes.data['h']
        h_agg = nodes.data['h_agg']
        deg = nodes.data['deg'].unsqueeze(1)

        # h_concat = nd.concat(h, h_agg / nd.maximum(deg, 1e-6), dim=1)     ### 矩阵拼接，https://blog.csdn.net/zhaoyunduan1958/article/details/107390797/
        # h_concat = th.concat(h, h_agg / th.maximum(deg, 1e-6), dim=1)
        p=th.tensor(1e-6)
        h_concat = th.cat([h, h_agg / th.maximum(deg, p)], 1)
        # h_new = self.dropout(self.leakyrelu(self.W(h_concat)))
        with torch.no_grad():
            h_new = self.dropout(self.leakyrelu(self.W(h_concat)))

        return {'h': h_new}
