import torch
from tqdm import tqdm
import torch.nn.functional as F
from model import CENN, CENN_1, LIN

def train(data, num_features, num_classes, communities, comm_feat, community_explain_flatten, args):
    model = CENN_1(num_features, num_classes, community_explain_flatten, args).to(torch.device(args.device))
    lin = LIN(num_features, args).to(torch.device(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_lin = torch.optim.Adam(lin.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    min_loss = 1e10
    max_acc = 0
    for epoch in range(args.epochs):
        new_feat = torch.randn((communities.size, args.nhid)).to(torch.device(args.device))
        optimizer.zero_grad()
        for i in range(communities.size):
            new_feat[i] = lin(comm_feat[i].to(torch.device(args.device)))

        out = model(data.x, data.edge_index, new_feat)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer_lin.step()
        if epoch % 10 == 0:
            val_acc, val_loss = eval(model, lin, data, communities, comm_feat, args)
            print(f"epoch:{epoch}, loss{loss}\tval acc:{val_acc}")
            if val_loss < min_loss:
                torch.save(model.state_dict(), f'check/train_latest_great_{args.dataset}.pth')
                print("train_model saved at epoch_", epoch)
                min_loss = val_loss

            if val_acc > max_acc:
                torch.save(model.state_dict(), f'check/train_latest_great_{args.dataset}.pth')
                print("train_model saved at epoch_", epoch)
                max_acc = val_acc

    model.load_state_dict(torch.load(f'check/train_latest_great_{args.dataset}.pth'))
    return model, lin


def eval(model, lin, data, communities, comm_feat, args):
    model.eval()

    loss = 0
    num_count = 0
    new_feat = torch.randn((communities.size, args.nhid)).to(torch.device(args.device))
    for i in range(communities.size):
        new_feat[i] = lin(comm_feat[i].to(torch.device(args.device)))

    out = model(data.x, data.edge_index, new_feat)
    pred = out.max(dim=1)[1]
    loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    acc = correct / int(data.val_mask.sum())

    return acc, loss


def test(model, lin, data, communities, comm_feat, args):
    model.eval()

    new_feat = torch.randn((communities.size, args.nhid)).to(torch.device(args.device))
    for i in range(communities.size):
        new_feat[i] = lin(comm_feat[i].to(torch.device(args.device)))

    out = model(data.x, data.edge_index, new_feat)
    pred = out.max(dim=1)[1]
    loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.val_mask.sum())

    return acc, loss