from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import args

prog_args = args.arg_parse()

num_epoch = prog_args.epochs
num_filter = prog_args.num_filter
max_nodes = 128  # ////////////////////////
dataset = prog_args.dataset

if dataset == '10a':
    num_classes = 11
elif dataset == '10b':
    num_classes = 10
else:
    print('ERROR')


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super(GNN, self).__init__()
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv4 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv5 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn5 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv6 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn6 = torch.nn.BatchNorm1d(out_channels)
        if lin is True:
            self.lin = torch.nn.Linear(5 * hidden_channels + out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size_, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size_, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))
        x4 = self.bn(4, F.relu(self.conv4(x3, adj, mask)))
        x5 = self.bn(5, F.relu(self.conv5(x4, adj, mask)))
        x6 = self.bn(6, F.relu(self.conv6(x5, adj, mask)))
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=-1)
        if self.lin is not None:
            x = F.relu(self.lin(x))
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.num_filter = num_filter  # self.num_filter决定了对角线的条数
        # 用于得到对角线相关数据
        self.add_conv1d = nn.Sequential()
        for i in range(self.num_filter):
            self.add_conv1d.add_module('new_conv1d_{}'.format(i + 2),
                                       nn.Conv1d(in_channels=1, out_channels=1, kernel_size=i + 2))
        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(2, 64, num_nodes)  # #############################################
        self.gnn1_embed = GNN(2, 64, 64, lin=False)  # #############################################
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(6 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(6 * 64, 64, 64, lin=False)
        self.gnn3_embed = GNN(6 * 64, 64, 64, lin=False)
        self.lin1 = torch.nn.Linear(6 * 64, 3 * 64)
        # self.dropout1 = torch.nn.Dropout(0.5)
        self.lin2 = torch.nn.Linear(3 * 64, 64)
        # self.dropout2 = torch.nn.Dropout(0.5)
        self.lin3 = torch.nn.Linear(2 * 64, 64)
        # self.dropout3 = torch.nn.Dropout(0.5)
        self.lin4 = torch.nn.Linear(64, num_classes)

    def forward(self, x_in, mask=None):  # [batch_size, 1, length, 2]
        # 处理I通道数据。得到邻接矩阵（其实是特征矩阵）
        adj_i = torch.zeros(x_in.shape[0], x_in.shape[2], x_in.shape[2]).cuda()  # (batch_size, length, length)
        x_in_i = x_in[:, :, :, 0]  # [batch_size, 1, length]
        x_in_q = x_in[:, :, :, 1]  # [batch_size, 1, length]
        for i in range(self.num_filter):
            other_x = self.add_conv1d[i](x_in_i)  # [batch_size, 1, length+1-(i+2)]，一维卷积处理后的序列
            other_x = torch.squeeze(other_x)  # [batch_size, length+1-(i+2)]
            other_x = F.relu(other_x, inplace=True)  # [batch_size, length+1-(i+2)]
            other_x = torch.diag_embed(other_x, offset=(i + 1))  # [batch_size, length, length]
            adj_i = torch.add(adj_i, other_x)  # 各个对角线矩阵依次相加[batch_size, length, length]
        adj_i_2 = adj_i.permute(0, 2, 1)
        adj_i = adj_i + adj_i_2
        # 节点属性特征
        x_i = x_in_i.permute(0, 2, 1)  # [batch_size, length, 1]
        x_q = x_in_q.permute(0, 2, 1)  # [batch_size, length, 1]
        x_iq = torch.cat((x_i, x_q), 2) * 10  # [batch_size, length, 2]

        x_i = x_iq  # [batch_size, length, 2]
        s_i = self.gnn1_pool(x_i, adj_i, mask)
        x_i = self.gnn1_embed(x_i, adj_i, mask)
        x_i, adj_i, l1, e1 = dense_diff_pool(x_i, adj_i, s_i, mask)
        s_i = self.gnn2_pool(x_i, adj_i)
        x_i = self.gnn2_embed(x_i, adj_i)
        x_i, adj_i, l2, e2 = dense_diff_pool(x_i, adj_i, s_i)
        x_i = self.gnn3_embed(x_i, adj_i)
        x_i = x_i.mean(dim=1)
        x_i = F.relu(self.lin1(x_i))
        # x_i = self.dropout1(x_i)
        x_i = F.relu(self.lin2(x_i))
        # x_i = self.dropout2(x_i)
        # 处理Q通道数据。得到邻接矩阵（其实是特征矩阵）
        adj_q = torch.zeros(x_in.shape[0], x_in.shape[2], x_in.shape[2]).cuda()  # (batch_size, length, length)
        x_in_q = x_in[:, :, :, 1]  # [batch_size, 1, length]
        for q in range(self.num_filter):
            other_x_ = self.add_conv1d[q](x_in_q)  # [batch_size, 1, length+1-(i+2)]，一维卷积处理后的序列
            other_x_ = torch.squeeze(other_x_)  # [batch_size, length+1-(i+2)]
            other_x_ = F.relu(other_x_, inplace=True)  # [batch_size, length+1-(i+2)]
            other_x_ = torch.diag_embed(other_x_, offset=(q + 1))  # [batch_size, length, length]
            adj_q = torch.add(adj_q, other_x_)  # 各个对角线矩阵依次相加[batch_size, length, length]
        adj_q_2 = adj_q.permute(0, 2, 1)
        adj_q = adj_q + adj_q_2
        # 节点属性特征
        x_q = x_iq  # [batch_size, length, 2]
        s_q = self.gnn1_pool(x_q, adj_q, mask)
        x_q = self.gnn1_embed(x_q, adj_q, mask)
        x_q, adj_q, l1, e1 = dense_diff_pool(x_q, adj_q, s_q, mask)
        s_q = self.gnn2_pool(x_q, adj_q)
        x_q = self.gnn2_embed(x_q, adj_q)
        x_q, adj_q, l2, e2 = dense_diff_pool(x_q, adj_q, s_q)
        x_q = self.gnn3_embed(x_q, adj_q)
        x_q = x_q.mean(dim=1)
        x_q = F.relu(self.lin1(x_q))
        # x_q = self.dropout1(x_q)
        x_q = F.relu(self.lin2(x_q))
        # x_q = self.dropout2(x_q)
        # 融合IQ通道得到的特征
        x = torch.cat((x_i, x_q), 1)

        x = F.relu(self.lin3(x))
        # x = self.dropout3(x)
        x = self.lin4(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2