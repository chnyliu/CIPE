import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import *
from network_arma import ARMAConv_
from torch.nn import Sequential, Linear, ReLU
from network_geolayer import GeoLayer
from torch_geometric.nn.models import JumpingKnowledge
import math
from utils import arch_simple


def act_map(act):
    if act == "elu":
        return F.elu
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


def gnn_map(gnn_name, in_dim, out_dim, dropout, concat=True, bias=True):
    """
    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param dropout:
    :param concat: for gat, concat multi-head output or not
    :param bias:
    :return: GNN model
    """

    if gnn_name == "gat_8":
        return GATConv(in_dim, out_dim, 8, concat=False, bias=bias, dropout=dropout)
    elif gnn_name == "gat_4":
        return GATConv(in_dim, out_dim, 4, concat=False, bias=bias, dropout=dropout)
    elif gnn_name == "gat_1":
        return GATConv(in_dim, out_dim, 1, concat=False, bias=bias, dropout=dropout)
    elif gnn_name == "gcn":
        return GCNConv(in_dim, out_dim)
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "sage_mean":
        return SAGEConv(in_dim, out_dim, bias=bias, aggr='mean')
    elif gnn_name == "sage_max":
        return SAGEConv(in_dim, out_dim, bias=bias, aggr='max')
    elif gnn_name == "sage_sum":
        return SAGEConv(in_dim, out_dim, bias=bias, aggr='add')
    elif gnn_name == "arma":
        return ARMAConv_(in_dim, out_dim, bias=bias)
    elif gnn_name == "appnp":
        return APPNP(K=10, alpha=0.1)
    elif gnn_name == 'gin':
        nn1 = Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
        return GINConv(nn1)
    elif gnn_name in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
        head_num = 8
        return GeoLayer(in_dim, int(out_dim / head_num), heads=head_num, att_type=gnn_name, dropout=dropout,
                        concat=concat)
    else:
        raise Exception("wrong gnn name")


class GraphEhanced(nn.Module):
    def __init__(self, out_dim, residual=False):
        super(GraphEhanced, self).__init__()

        self.out_dim = out_dim
        self.residual = residual

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_dim)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_1.data.uniform_(-stdv, stdv)

    def forward(self, last_input, h0):
        assert last_input.size() == h0.size()

        mixed = last_input + h0
        output = mixed

        return output


class Cell(nn.Module):
    def __init__(self, action_list, num_feat, hidden_dim, cell_index, dropout, concat, bias=True):
        super(Cell, self).__init__()
        self.action_list = action_list
        self.num_feat = num_feat
        self.hidden_dim = hidden_dim
        self.cell_index = cell_index
        self.dropout = dropout
        self.concat_of_multihead = concat
        self.bias = bias

        self._indices = []
        self._layer_agg_list = []
        self._layer_agg_lstm_index = []
        self._layer_agg_lstm = nn.ModuleList()
        self._act_list = []
        self._appnp_layer_index = []

        if self.cell_index == 0:
            self.layer0 = nn.Linear(self.num_feat, self.hidden_dim)
        self._gnn_aggrs = nn.ModuleList()
        self._appnp_trans = nn.ModuleList()
        self._enhanced_gnn = nn.ModuleList()

        self._compile()
        self.reset_parameters()

    def reset_parameters(self):
        for aggr in self._gnn_aggrs:
            aggr.reset_parameters()

    def _compile(self):
        self._concat = self.action_list[-1]

        cells_info = self.action_list[:-1]
        cells_info_simple = arch_simple(cells_info)

        assert len(cells_info_simple) % 4 == 0
        self.real_num_cell_layers = len(cells_info_simple) // 4
        in_dim = self.hidden_dim

        for i, action in enumerate(cells_info_simple):
            if i % 4 == 0:
                self._indices.append(action)
            elif i % 4 == 1:
                self._layer_agg_list.append(action)
                if action == 'concat':
                    in_dim = sum(self._indices[-1]) * self.hidden_dim
                elif action == 'lstm':
                    self._layer_agg_lstm_index.append(int((i - 1) / 4))
                    self._layer_agg_lstm.append(JumpingKnowledge(mode='lstm', channels=self.hidden_dim, num_layers=2))
            elif i % 4 == 2:
                premiere_gnn = gnn_map(action, in_dim, self.hidden_dim, self.dropout, self.concat_of_multihead,
                                       self.bias)
                enhanced_gnn = GraphEhanced(self.hidden_dim)

                self._gnn_aggrs.append(premiere_gnn)
                if action == 'appnp' and in_dim != self.hidden_dim:
                    self._appnp_trans.append(nn.Linear(in_dim, self.hidden_dim))
                    self._appnp_layer_index.append(int((i - 2) / 4))
                self._enhanced_gnn.append(enhanced_gnn)

                in_dim = self.hidden_dim
            else:
                self._act_list.append(act_map(action))

        if self._concat == 'lstm':
            self.jk_func = JumpingKnowledge(mode='lstm', channels=self.hidden_dim, num_layers=2).cuda()
        elif self._concat == 'concat':
            self.jk_func = Linear((2 + self.real_num_cell_layers) * self.hidden_dim, self.hidden_dim)

    def forward(self, s0, s1, edge_index, h0):
        if self.cell_index == 0:
            s0 = self.layer0.forward(s0)
        features = [s0, s1]
        for i in range(self.real_num_cell_layers):
            indices = self._indices[i]
            h1 = []
            for j in range(i + 2):
                if indices[j] == 1:
                    assert len(features[j]) != 0
                    h1.append(features[j])

            layer_agg_type = self._layer_agg_list[i]
            if layer_agg_type == 'max':
                h1 = torch.stack(h1, dim=0)
                h1 = torch.max(h1, dim=0)[0]
            elif layer_agg_type == 'add':
                h1 = torch.stack(h1, dim=0)
                h1 = torch.sum(h1, dim=0)
            elif layer_agg_type == 'concat':
                h1 = torch.cat(h1, dim=1)
            else:
                assert i in self._layer_agg_lstm_index
                h1 = self._layer_agg_lstm[self._layer_agg_lstm_index.index(i)](h1)

            op1 = self._gnn_aggrs[i]
            op2 = self._enhanced_gnn[i]
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            s_premiere = op1(h1, edge_index)
            if i in self._appnp_layer_index:
                op_appnp = self._appnp_trans[self._appnp_layer_index.index(i)]
                s_premiere = op_appnp(s_premiere)
            s_enhanced = self._act_list[i](op2(s_premiere, h0))
            features.append(s_enhanced)
        if self._concat == 'max':
            out = torch.stack(features, dim=0)
            out = torch.max(out, dim=0)[0]
        elif self._concat == 'add':
            out = torch.stack(features, dim=0)
            out = torch.sum(out, dim=0)
        elif self._concat == 'concat':
            out = torch.cat(features, dim=1)
            out = self.jk_func(out)
        else:
            out = self.jk_func(features)
        return out


class GNN(nn.Module):
    def __init__(self, action, num_feat, num_classes, num_hidden, dropout, num_cells, bias=True):
        super(GNN, self).__init__()
        self.action = action
        self.num_feat = num_feat
        self.num_classes = num_classes
        self.hidden_dim = num_hidden
        self.num_cells = num_cells
        self.dropout = dropout
        self.bias = bias

        assert len(self.action[:-1]) % 4 == 0
        self.num_cell_layers = len(self.action[:-1]) // 4

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(self.num_feat, self.hidden_dim))

        self.cells = nn.ModuleList()

        for i in range(num_cells):
            cell = Cell(self.action, self.num_feat, self.hidden_dim, cell_index=i, dropout=self.dropout, concat=True,
                        bias=bias)
            self.cells += [cell]

        self.fcs.append(nn.Linear(self.hidden_dim, self.num_classes))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        s0 = x
        s1 = self.fcs[0](x)

        h0 = s1

        for _, cell in enumerate(self.cells):
            s2 = cell(s0, s1, edge_index, h0)
            s0 = s1
            s1 = s2
        out = s1
        logits = self.fcs[-1](out.view(out.size(0), -1))
        return logits
