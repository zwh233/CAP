import random
import numpy as np
from tqdm import tqdm
from torch_geometric.datasets import Planetoid, CoraFull, CitationFull, Coauthor, Amazon, Flickr, Reddit, Reddit2, \
    GitHub, NELL, WikiCS, Twitch, FacebookPagePage, PolBlogs, NELL, MyketDataset
from torch_geometric.utils import to_networkx, degree, k_hop_subgraph, dropout_adj, to_dense_adj
# from communities.algorithms import louvain_method, spectral_clustering, girvan_newman
import time
import logging
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import networkx as nx
from networkx.algorithms.community import k_clique_communities, label_propagation_communities

def load_dataset(args):
    if args.dataset == 'cora':
        dataset = Planetoid(root='../dataset', name=args.dataset)
    if args.dataset == 'citeseer':
        dataset = Planetoid(root='../dataset', name='citeseer')
    if args.dataset == 'pubmed':
        dataset = Planetoid(root='../dataset', name='PubMed')
    if args.dataset == "corafull":
        dataset = CoraFull(root="../dataset/Corafull")
    if args.dataset == 'photo':
        dataset = Amazon(root='../dataset', name='Photo')
    if args.dataset == 'computers':
        dataset = Amazon(root='../dataset', name='Computers')
    if args.dataset == 'physics':
        dataset = Coauthor(root='../dataset', name='Physics')
    if args.dataset == 'cs':
        dataset = Coauthor(root='../dataset', name='CS')
    if args.dataset == 'dblp':
        dataset = CitationFull(root='../dataset', name='DBLP')
    if args.dataset == 'flickr':
        dataset = Flickr(root='../dataset/Flickr')
    if args.dataset == 'reddit':
        dataset = Reddit(root='../dataset/Reddit')
    if args.dataset == 'nell':
        dataset = NELL(root='../dataset/NELL')
    if args.dataset == 'myket':
        dataset = MyketDataset(root='../dataset/Myket')
    if args.dataset == 'ploblogs':
        dataset = PolBlogs(root='../dataset/PolBlogs')
        dataset.data.x = torch.eye(dataset.data.num_nodes)

    data = dataset[0]
    num_node_features = dataset.num_node_features
    data = random_split(args, data)
    return data, num_node_features, dataset.num_classes


def random_split(args, data):
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)

    train_idx = np.random.choice(data.num_nodes, int(data.num_nodes * args.train_ratio), replace=False)
    residue = np.array(list(set(range(data.num_nodes)) - set(train_idx)))
    val_idx = np.random.choice(residue, int(data.num_nodes * args.val_ratio), replace=False)
    test_idx = np.array(list(set(residue) - set(val_idx)))

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def load_communities(data, args):
    # adj = to_dense_adj(data.edge_index)
    if os.path.exists(f'./data_preprocessed/{args.dataset}_communities.npy'):
        print(f'Loading {args.dataset}_communities ...')
        communities = np.load(f'./data_preprocessed/{args.dataset}_communities.npy', allow_pickle=True)
    else:
        T1 = time.time()
        print(f'Construct {args.dataset}_communities ...')
        # communities, _ = louvain_method(adj.squeeze(), 14)
        # communities = girvan_newman(adj.squeeze().numpy(), 20)
        G = trans_to_networkx(data)
        # communities = list(louvain_communities(G))
        communities = list(label_propagation_communities(G))
        np.save(f'./data_preprocessed/{args.dataset}_communities.npy', communities)
        T2 = time.time()
        print(f'Construct cost {(T2 - T1) * 1000 / 3600} hours')
    print(f'Loading {args.dataset}_communities successful')

    return communities


def get_communities_features(communities, data):
    community = list()
    for comm in tqdm(range(communities.size)):
        temp = list()
        for v in list(communities[comm]):
            temp.append(data.x[v])
        # temp = torch.tensor([item.cpu().detach().numpy() for item in temp]).sum(axis=0)
        temp = torch.tensor([item.cpu().detach().numpy() for item in temp])
        community.append(temp)

    # return torch.tensor([item.cpu().detach().numpy() for item in community])
    return community


def get_communities_features_tensor(communities, data):
    community = list()
    for comm in tqdm(range(len(communities))):
        temp = list()
        for v in list(communities[comm]):
            temp.append(data.x[v])
        temp = torch.tensor([item.cpu().detach().numpy() for item in temp]).sum(axis=0)
        # temp = torch.tensor([item.cpu().detach().numpy() for item in temp])
        community.append(temp)

    return torch.tensor([item.cpu().detach().numpy() for item in community])
    # return community


def concat_feat(x, comm_feat, num_features, communities, nhid):
    x1 = torch.randn((x.shape[0], num_features + nhid)).to(x.device)
    # x1 = torch.randn((x.shape[0], 2 * num_features)).to(x.device)

    for comm in range(communities.size):
        index = list(communities[comm])
        x1[index] = torch.cat((x[index][:, :num_features], comm_feat[comm].repeat(len(index), 1)), dim=-1)

    return x1


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def visualization(out):
    tsne = TSNE(n_components=2)
    tsne.fit_transform(out)
    return tsne.embedding_


def Print(tsne, args, y):
    fig = plt.figure(figsize=(8, 8))
    label = y.cpu().detach().numpy()
    plt.scatter(tsne[:, 0], tsne[:, 1], c=label, s=30, cmap=plt.cm.RdYlBu)

    # # 定义颜色RGB值，可以根据需要调整以获得更浅的颜色
    # colors = [(234 / 255, 87 / 255, 57 / 255),  # 浅红色
    #           (254 / 255, 224 / 255, 144 / 255),  # 浅橙色
    #           (99 / 255, 153 / 255, 199 / 255)]
    #
    # # 创建一个线性分段的colormap
    # cmap_name = 'my_custom_cmap'
    # n_bins = 100  # 分段数量，可以根据需要调整
    # from matplotlib.colors import LinearSegmentedColormap
    # cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    #
    #
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=label, s=30, cmap=cmap)

    # plt.title(f"{args.dataset}")
    plt.axis('off')
    # plt.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # plt.yaxis.set_major_formatter(NullFormatter())                                                                        ccccccdfvvbgvfbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbhnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnjmmmmmmmkjk,m,,,,,,,,,,,,、、、、、、、、、、、


    plt.show()

    fig.savefig(f'{args.dataset}.pdf', dpi=600, format='pdf', bbox_inches='tight')


def trans_to_networkx(data):
    G = nx.Graph()

    # 使用 add_nodes_from 批处理的效率比 add_node 高
    G.add_nodes_from([i for i in range(data.num_nodes)])

    # 使用 add_edges_from 批处理的效率比 add_edge 高
    edges = np.array(data.edge_index.T, dtype=int)
    G.add_edges_from(edges)

    return G


def Cluster_comm(communities, comm_feat, n_clusters):
    tsne = TSNE(n_components=2, perplexity=25, n_iter=2000, early_exaggeration=50)
    tsne.fit_transform(comm_feat)

    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(tsne.embedding_)
    new_communities = list()
    for i in range(n_clusters):
        new_communities.append([])

    for iter, comm in enumerate(cluster.labels_):
        new_communities[comm].append(communities[iter])

    for i in range(n_clusters):
        new_communities[i] = [j for i in new_communities[i] for j in i]

    return new_communities


def get_top_nodes_by_degree(community, graph, n):
    node_degrees = {node: graph.degree(node) for node in community}
    sorted_nodes = sorted(node_degrees, key=lambda node: node_degrees[node], reverse=True)
    top_nodes = sorted_nodes[:n]
    return top_nodes


def generate_subgraph(start_node, community, graph, subgraph_size):
    subgraph_nodes = [start_node]
    while len(subgraph_nodes) < subgraph_size:

        best_node = None
        max_modularity_decrease = -float('inf')

        community_modularity = compute_modularity(community, graph)

        for node in community:
            if node not in subgraph_nodes and any(graph.has_edge(node, n) for n in subgraph_nodes):
                candidate_nodes = subgraph_nodes + [node]
                new_community = community.difference(set(candidate_nodes))
                modularity_decrease = community_modularity - compute_modularity(new_community, graph)
                if modularity_decrease > max_modularity_decrease:
                    max_modularity_decrease = modularity_decrease
                    best_node = node

        if best_node is None:
            break

        subgraph_nodes.append(best_node)

    return subgraph_nodes


def compute_modularity(nodes, graph):
    subgraph = graph.subgraph(nodes)  # 获取子图

    # 创建子图的邻接矩阵
    adjacency_matrix = nx.adjacency_matrix(subgraph).toarray()
    adjacency_matrix = adjacency_matrix.astype(float)

    # 计算子图的度矩阵
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

    # 计算子图的模块度
    modularity = nx.modularity_matrix(subgraph)
    modularity_score = np.trace(adjacency_matrix) - np.trace(np.dot(degree_matrix, modularity))

    return modularity_score


def draw_subgraph(subgraph_nodes, graph):
    subgraph = graph.subgraph(subgraph_nodes)
    pos = nx.spring_layout(subgraph)  # 定义节点布局

    # 绘制子图的节点和边
    nx.draw_networkx_nodes(subgraph, pos, node_color='r', node_size=20)
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5)

    # 添加节点标签
    # labels = {node: node for node in subgraph.nodes()}
    # nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)

    plt.axis('off')  # 关闭坐标轴显示
    plt.show()


def flatten(a):
    for each in a:
        if not isinstance(each, list):
            yield each
        else:
            yield from flatten(each)


def plot_subgraph(G, node):
    # 找到所有一跳邻居
    one_hop_neighbors = set(G.neighbors(node))

    # 初始化二跳邻居集合
    two_hop_neighbors = set()

    # 遍历一跳邻居，找到它们的邻居（即二跳邻居）
    for neighbor in one_hop_neighbors:
        two_hop_neighbors.update(G.neighbors(neighbor))

    # 从二跳邻居中移除一跳邻居和节点本身，以确保只有二跳邻居
    two_hop_neighbors.difference_update(one_hop_neighbors)
    two_hop_neighbors.discard(node)

    # 包括目标节点、一跳邻居和二跳邻居
    node_list = [node] + list(one_hop_neighbors | two_hop_neighbors)

    # 从节点列表创建子图
    subgraph = G.subgraph(node_list).copy()

    # 找到所有简单路径，按路径长度排序
    all_paths = list(nx.shortest_simple_paths(G, source=1115, target=868))

    # 选择前三条最短的路径
    top_paths = all_paths[:3] if len(all_paths) >= 3 else all_paths

    # 画出子图
    pos = nx.spring_layout(subgraph)  # 所有节点的位置
    options = {
        "node_color": "lightblue",
        "node_size": 50,
        "linewidths": 2,
        "width": 1,
        "with_labels": True,  # 设置为True在节点上画标签
    }
    nx.draw_networkx(subgraph, pos, **options)

    # 显示图形
    plt.show()


def sample_positive_indices(data, communities, community_explain_flatten=None, batch_id=None, sample_len=5):
    train_mask = data.train_mask
    num_nodes = data.num_nodes

    positive_node_indices = []

    for node in range(num_nodes):
        if train_mask[node]:
        # if True:
            community_index = find_community_index(communities, node)
            node_label = data.y[node]
            if community_explain_flatten is not None and community_index < len(community_explain_flatten):
                community_nodes = community_explain_flatten[community_index]
            else:
                community_nodes = communities[community_index]
            # Find nodes with the same label within the community
            # same_label_nodes = [n for n in community_nodes if train_mask[n] and data.y[n] == node_label]
            same_label_nodes = [n for n in community_nodes]
            if len(same_label_nodes) == 0:
                # Remove the condition data.y[n] == node_label and construct same_label_nodes again
                same_label_nodes = [n for n in community_nodes]
                if len(same_label_nodes) == 0:
                    # If same_label_nodes is still empty, include only the node itself
                    same_label_nodes = [node]
            if len(same_label_nodes) < sample_len:
                # If there are fewer than 5 nodes with the same label in the community, duplicate them
                sampled_nodes = same_label_nodes * (sample_len // len(same_label_nodes))
                remaining = sample_len % len(same_label_nodes)
                sampled_nodes.extend(random.sample(same_label_nodes, k=remaining))
            else:
                # Randomly sample 5 nodes with the same label from the community
                sampled_nodes = random.sample(same_label_nodes, k=sample_len)
            positive_node_indices.extend(sampled_nodes)

    return positive_node_indices


def sample_negative_indices(data, communities, community_explain_flatten=None, batch_id=None, sample_len=5):
    train_mask = data.train_mask
    num_nodes = data.num_nodes

    negative_node_indices = []

    for node in range(num_nodes):
        if train_mask[node]:
        # if True:
             community_index = find_community_index(communities, node)
             node_label = data.y[node]
             negative_community_indices = []
             if community_explain_flatten is not None and community_index < len(community_explain_flatten):
                 community_nodes = community_explain_flatten[community_index]
             else:
                 community_nodes = communities[community_index]
             # Find communities with different labels
             for i, community in enumerate(communities):
                 # if i != community_index and has_different_labels(data, community, node_label):
                 if i != community_index:
                     negative_community_indices.append(i)
             if len(negative_community_indices) > 0:
                 # Randomly sample 5 nodes from different communities with different labels
                 for _ in range(sample_len):
                     random_community_index = random.choice(negative_community_indices)
                     if community_explain_flatten is not None and random_community_index < len(
                             community_explain_flatten):
                         random_node = random.choice(list(community_explain_flatten[random_community_index]))
                     else:
                         random_node = random.choice(list(communities[random_community_index]))
                     negative_node_indices.append(random_node)
             else:
                 # If no other communities have different labels, copy the node itself as negative samples
                 negative_node_indices.extend([node] * sample_len)

    return negative_node_indices


def get_selected_features(data, indices):
    selected_features = data[indices]
    num_samples = len(indices)
    num_features = data.size(1)
    selected_features = selected_features.view(num_samples, num_features)
    selected_features = selected_features.view(-1, 5, num_features)
    return selected_features


def get_comm_labels(communities, data):
    comm_labels = [-1 for _ in range(data.num_nodes)]
    for node in range(data.num_nodes):
        comm_labels[node] = find_community_index(communities, node)

    return comm_labels

def find_community_index(communities, node):
    for i, community in enumerate(communities):
        if node in community:
            return i

    return -1


def has_different_labels(data, community, node_label):
    for node in community:
        if data.y[node] != node_label:
            return True

    return False

def find_intersection(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    intersection = set1.intersection(set2)
    return list(intersection)