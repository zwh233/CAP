import torch
from utils import trans_to_networkx, get_communities_features_tensor, Cluster_comm, flatten, load_communities, load_dataset, sample_positive_indices, sample_negative_indices, get_selected_features
import argparse
from model import GCN, GAT, GraphSage, SplineCNN, AntiSymmetric, FusedGAT, Chebnet, ARMA, UniMP
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
from my_mcts import community_subgraph_explorer
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', help='dataset')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--nhid', type=int, default=128, help='nhid')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
parser.add_argument('--epochs', type=int, default=300, help='epochs')
parser.add_argument('--pre_train_epochs', type=int, default=20, help='epochs')
parser.add_argument('--seed', type=int, default=40, help="random seed")
parser.add_argument('--range', type=int, default=5, help="range")
parser.add_argument('--train_ratio', type=float, default=0.6, help='dataset')
parser.add_argument('--val_ratio', type=float, default=0.2, help='dataset')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')

parser.add_argument('--use_MCTS', type=bool, default=True, help='use_MCTS')
parser.add_argument('--num_iterations', type=int, default=100, help='Iterations of MCTS')
parser.add_argument('--subgraph_density', type=int, default=50, help='Generate one subgraph for subgraph_density node')
parser.add_argument('--subgraph_scale', type=int, default=15, help='The number of nodes in subgraph')


args = parser.parse_args()

def main():
    data, num_features, num_classes = load_dataset(args)
    communities = load_communities(data, args)

    G = trans_to_networkx(data)

    community_subgraph = community_subgraph_explorer(G, communities, args)
    community_subgraph_flatten = np.array([set(list(flatten(comm))) for comm in community_subgraph])

    data.to(torch.device(args.device))
    acc_list = []
    for i in range(args.range):
        print(f'-------------------------range{i}-------------------------')
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(args.seed)

        model = GCN(num_features, num_classes, args).to(torch.device(args.device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        positive_indices = sample_positive_indices(data, communities, community_subgraph_flatten)
        negative_indices = sample_negative_indices(data, communities, community_subgraph_flatten)
        triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

        model.train()
        acc_best = []
        for epoch in tqdm(range(args.epochs)):
            optimizer.zero_grad()
            if epoch < args.pre_train_epochs:
            # if False:
                out = model(data.x, data.edge_index)
                anchor_features = out[data.train_mask].unsqueeze(1).expand(-1, 5, -1)
                positive_features = get_selected_features(out, positive_indices)
                negative_features = get_selected_features(out, negative_indices)
                community_aware_loss = triplet_loss(anchor_features, positive_features, negative_features)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                if epoch % 10 == 0:
                    print(f"epoch:{epoch}, community_aware_loss{community_aware_loss:.4f}, loss{loss:.4f}")
                community_aware_loss.backward()
            else:
                out = model(data.x, data.edge_index)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                if epoch % 10 == 0:
                    test_acc = evaluate_model(model, data.x, data, data.test_mask)
                    val_acc = evaluate_model(model, data.x, data, data.val_mask)
                    print(f"epoch:{epoch}, loss{loss:.4f}, Test Accuracy: {test_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
                    acc_best.append(test_acc)
            optimizer.step()


        print(f"Best Accuracy: {max(acc_best):.4f}")

        acc_list.append(max(acc_best))

    print(f"mean:{np.mean(acc_list)}")
    print(f"max:{max(acc_list)}")
    print(f"min:{min(acc_list)}")
    print(f"std:{np.std(acc_list)}")

    # logger.info(f"mean:{np.mean(acc_list)}, max:{max(acc_list)}, min:{min(acc_list)}, std:{np.std(acc_list)}")

def evaluate_model(model, x, data, mask):
    model.eval()
    _, pred = model(x, data.edge_index).max(dim=1)
    correct = int(pred[mask].eq(data.y[mask]).sum().item())
    acc = correct / int(mask.sum())
    return acc

if __name__ == '__main__':
    print('---------args-----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------start!----------\n')

    main()

    print('---------args-----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------end!----------\n')
