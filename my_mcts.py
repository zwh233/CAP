from tqdm import tqdm
import random
import math
from utils import compute_modularity, generate_subgraph, get_top_nodes_by_degree
import os
import numpy as np
import time

def community_subgraph_explorer(G, communities, args):
    community_subgraphs = []
    t1 = time.time()
    if args.use_MCTS:
        if not os.path.exists(
                f'./subgraph/MCTS_iter{args.num_iterations}_{args.dataset}_{args.subgraph_density}density_{args.subgraph_scale}scale.npy'):
            print(f'Construct {args.dataset}_community subgraph.')

            for i, community in tqdm(enumerate(communities)):
                community_subgraphs.append([])
                print(' ')
                print(f'-------------------------Community:{i} / {len(communities)}-------------------------')
                if len(community) <= args.subgraph_scale:
                    print(f'Community:{i} only have {len(community)} nodes.')
                    community_subgraphs[i].append(list(community))
                    continue

                # top_nodes = get_top_nodes_by_degree(community, G, int(len(community) / args.subgraph_density) if len(
                #     community) > args.subgraph_density else 1)
                top_nodes = get_top_nodes_by_degree(community, G, 1)

                for start_node in top_nodes:
                    print(f'----------Root:{start_node} / {top_nodes}----------')
                    subgraph_scale = len(community) / 10 if len(community) / 10 > args.subgraph_scale else args.subgraph_scale
                    subgraph_nodes = find_max_modularity_subgraph(start_node, G, community, subgraph_scale, args.num_iterations)
                    community_subgraphs[i].append(subgraph_nodes)
                    # draw_subgraph(subgraph_nodes, G)

            np.save(f'./subgraph/MCTS_iter{args.num_iterations}_{args.dataset}_{args.subgraph_density}density_{args.subgraph_scale}scale.npy',
                    community_subgraphs)
            t2 = time.time()
            print(f'subgraph cost {(t2 - t1) * 1000 / 3600 :.4f} hours')
        else:
            community_subgraphs = np.load(
                f'./subgraph/MCTS_iter{args.num_iterations}_{args.dataset}_{args.subgraph_density}density_{args.subgraph_scale}scale.npy',
                allow_pickle=True)

    else:
        if not os.path.exists(f'./subgraph/{args.dataset}_{args.subgraph_density}density_{args.subgraph_scale}scale.npy'):
            print(f'Construct {args.dataset}_community subgraph.')

            for i, community in tqdm(enumerate(communities)):
                community_subgraphs.append([])
                print(' ')
                print(f'-------------------------Community:{i} / {len(communities)}-------------------------')
                if len(community) <= args.subgraph_scale:
                    print(f'Community:{i} only have {len(community)} nodes.')
                    community_subgraphs[i].append(list(community))
                    continue

                # top_nodes = get_top_nodes_by_degree(community, G, int(len(community) / args.subgraph_density) if len(
                #     community) > args.subgraph_density else 1)
                top_nodes = get_top_nodes_by_degree(community, G, 1)

                for start_node in top_nodes:
                    print(f'----------Root:{start_node} / {top_nodes}----------')
                    subgraph_scale = len(community) / 10 if len(community) / 10 > args.subgraph_scale else args.subgraph_scale
                    subgraph_nodes = generate_subgraph(start_node, community, G, subgraph_scale)
                    community_subgraphs[i].append(subgraph_nodes)
                    # draw_subgraph(subgraph_nodes, G)

            np.save(f'./subgraph/{args.dataset}_{args.subgraph_density}density_{args.subgraph_scale}scale.npy',
                    community_subgraphs)
            t2 = time.time()
            print(f'subgraph cost {(t2 - t1) * 1000 / 3600 :.4f} hours')
        else:
            community_subgraphs = np.load(
                f'./subgraph/{args.dataset}_{args.subgraph_density}density_{args.subgraph_scale}scale.npy',
                allow_pickle=True)
    return community_subgraphs


class MonteCarloTreeNode:
    def __init__(self, graph, community, subgraph, subgraph_scale, parent=None, node=None, root=False):
        self.graph = graph
        self.community = community
        self.subgraph = subgraph if subgraph is not None else []
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0.0
        self.node = node
        self.root = root
        self.subgraph_scale = subgraph_scale

    def expand(self):
        node = self.select_untried_node()
        child_subgraph = self.subgraph
        if node:
            child_subgraph.append(node)
        child = MonteCarloTreeNode(self.graph, self.community, child_subgraph, self.subgraph_scale, parent=self,
                                   node=node)
        self.children.append(child)
        return child

    def select_untried_node(self):
        tried_nodes = self.subgraph
        # untried_nodes = self.community - tried_nodes
        candidate_nodes = []
        for node in self.community:
            if node not in tried_nodes and any(self.graph.has_edge(node, n) for n in tried_nodes):
                candidate_nodes = candidate_nodes + [node]
        if len(candidate_nodes) > 0:
            return random.choice(list(candidate_nodes))
        else:
            return None

    def select_best_child(self, exploration_constant):
        best_child = None
        best_score = float("-inf")
        for child in self.children:
            if child.subgraph is not None:
                score = (child.score / compute_modularity(self.community,
                                                          self.graph)) / child.visits + exploration_constant * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
                if score > best_score:
                    best_child = child
                    best_score = score
        return best_child

    def backpropagate(self, score):
        self.visits += 1
        self.score += score
        if self.parent:
            self.parent.backpropagate(score)

    def is_fully_expanded(self):
        # all_nodes = self.community
        # tried_nodes = set(self.subgraph)
        # return all_nodes == tried_nodes
        return len(self.subgraph) >= self.subgraph_scale

    def is_leaf_node(self):
        return len(self.children) == 0 and self.root == False


def monte_carlo_tree_search(root, graph, community, num_iterations, exploration_constant):
    for _ in tqdm(range(num_iterations)):
        selected_node = tree_policy(root, exploration_constant)
        if not selected_node:
            continue
        score = calculate_score_by_modularity(graph, community, selected_node)
        selected_node.backpropagate(score)

    result = root
    while not result.is_leaf_node():
        if result.select_best_child(0.0):
            result = result.select_best_child(0.0)
        else:
            break
    return result.subgraph


def tree_policy(node, exploration_constant):
    while node and not node.is_leaf_node():
        if not node.is_fully_expanded():
            expanded_node = node.expand()
            if expanded_node:
                return expanded_node
        else:
            node = node.select_best_child(exploration_constant)
    return node


def default_policy(graph, community, node):
    subgraph = graph.subgraph(node.community)
    modularity = nx.algorithms.community.quality.modularity(graph, [community, set(subgraph.nodes)])
    return modularity


def find_max_modularity_subgraph(root, graph, community, subgraph_scale, num_iterations, exploration_constant=1.4):
    subgraph = [root]
    root_node = MonteCarloTreeNode(graph, community, subgraph, subgraph_scale, node=root, root=True)
    max_modularity_subgraph = monte_carlo_tree_search(
        root_node, graph, community, num_iterations, exploration_constant
    )
    return max_modularity_subgraph


def calculate_score_by_modularity(graph, community, mct_node):
    if mct_node is not None and not community == set(mct_node.subgraph):
        modularity_without_subgraph = compute_modularity(community - set(mct_node.subgraph), graph)
    else:
        modularity_without_subgraph = 0
    community_modularity = compute_modularity(community, graph)

    return community_modularity - modularity_without_subgraph


if __name__ == '__main__':
    from utils import trans_to_networkx, load_communities, load_dataset

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--nhid', type=int, default=128, help='nhid')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--epochs', type=int, default=250, help='epochs')
    parser.add_argument('--seed', type=int, default=40, help="random seed")
    parser.add_argument('--range', type=int, default=5, help="range")
    parser.add_argument('--train_ratio', type=float, default=0.6, help='dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='dataset')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')

    parser.add_argument('--num_iterations', type=int, default=200, help='Iterations of MCTS')
    parser.add_argument('--subgraph_density', type=int, default=30,
                        help='Generate one explanation for subgraph_density node')
    parser.add_argument('--subgraph_scale', type=int, default=15, help='The number of nodes in subgraph subgraph')

    args = parser.parse_args()

    import networkx as nx

    data, num_features, num_classes = load_dataset(args)
    communities = load_communities(data, args)

    G = trans_to_networkx(data)

    community = communities[23]
    # 设置根节点
    root = get_top_nodes_by_degree(community, G, int(len(community) / args.subgraph_density) if len(
        community) > args.subgraph_density else 1)[0]

    # 调用函数获取模块度影响最大的子图
    max_modularity_subgraph = find_max_modularity_subgraph(root, G, community, args.subgraph_scale, args.num_iterations)
    print("Max Modularity Subgraph:", max_modularity_subgraph)
    print(len(max_modularity_subgraph))
