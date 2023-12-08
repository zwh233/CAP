import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SplineConv,  Linear, AntiSymmetricConv, FusedGATConv, ChebConv, ARMAConv, TransformerConv
import torch.nn.functional as F
from torch.nn import Linear


class Chebnet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, args):
        super(Chebnet, self).__init__()
        self.conv1 = ChebConv(num_node_features, args.nhid, K=8)
        self.conv2 = ChebConv(args.nhid, num_classes, K=1)
        self.dropout = args.dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x1 = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x


class GCN_pre(torch.nn.Module):
    def __init__(self, num_node_features, nhid, args):
        super(GCN_pre, self).__init__()
        self.conv1 = GCNConv(num_node_features, nhid)
        self.conv2 = GCNConv(nhid, num_node_features)
        self.dropout = args.dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, args.nhid)
        self.conv2 = GCNConv(args.nhid, num_classes)
        self.dropout = args.dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x



class GraphSage(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, args):
        super(GraphSage, self).__init__()
        self.conv1 = SAGEConv(num_node_features, args.nhid)
        self.conv2 = SAGEConv(args.nhid, num_classes)
        self.dropout = args.dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x1 = x
        x = F.relu(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, args, head=8):
        super(GAT, self).__init__()
        self.gat1 = GATConv(num_node_features, args.nhid, head, dropout=args.dropout)
        self.gat2 = GATConv(args.nhid * head, num_classes, head, dropout=args.dropout)
        self.dropout = args.dropout

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)


class SplineCNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super().__init__()

        self.conv1 = SplineConv(num_features, args.nhid, dim=1, kernel_size=2)
        self.conv2 = SplineConv(args.nhid, num_classes, dim=1, kernel_size=2)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class AntiSymmetric(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super().__init__()
        self.hid_size = args.nhid
        self.dropout = args.dropout
        self.conv1 = AntiSymmetricConv(num_features)
        self.conv2 = AntiSymmetricConv(self.hid_size)
        self.l1 = Linear(num_features, self.hid_size)
        self.l2 = Linear(self.hid_size, num_classes)

    def forward(self, x, edge_index):
        # value = torch.rand(edge_index.size(1))
        # adj1 = to_torch_csc_tensor(edge_index, size=(x.shape[0], x.shape[0]))
        # adj2 = to_torch_csc_tensor(edge_index, value, size=(x.shape[0], x.shape[0]))

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x1 = x
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.l2(x)
        result = F.log_softmax(x, -1)
        return result, x1

    def final_features(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        return x


class FusedGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, args, gat_heads=8):
        super().__init__()
        self.hid_size = args.nhid
        self.dropout = 0
        self.conv1 = FusedGATConv(num_features, self.hid_size, heads=gat_heads, add_self_loops=False)
        self.conv2 = FusedGATConv(gat_heads * self.hid_size, num_classes, heads=1, add_self_loops=False)

    def forward(self, x, edge_index):
        csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(x.shape[0], x.shape[0]))
        x = self.conv1(x, csr, csc, perm)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, csr, csc, perm)
        result = F.log_softmax(x, -1)
        return result

    def final_features(self, x, edge_index):
        csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(x.shape[0], x.shape[0]))
        x = self.conv1(x, csr, csc, perm)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, csr, csc, perm)
        return x


class ARMA(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, args):
        super(ARMA, self).__init__()
        self.conv1 = ARMAConv(num_node_features, args.nhid)
        self.conv2 = ARMAConv(args.nhid, num_classes)
        self.dropout = args.dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        x = F.softmax(x, dim=1)

        return x


class UniMP(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, args):
        super(UniMP, self).__init__()
        self.conv1 = TransformerConv(num_node_features, args.nhid)
        self.conv2 = TransformerConv(args.nhid, num_classes)
        self.dropout = args.dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        x = F.softmax(x, dim=1)

        return x