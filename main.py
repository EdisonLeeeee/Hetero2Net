import argparse
from copy import copy

import torch
import torch_geometric.transforms as T

from torch_geometric import seed_everything
from torch_geometric.datasets import OGB_MAG, HGBDataset
from torch_geometric.loader import ImbalancedSampler, NeighborLoader
from torch_geometric.utils import sort_edge_index, index_to_mask

# custom modules
from logger import setup_logger
from models import HeteroGNN
from rcdd import RCDDataset
from utils import (evaluate_full_batch,
                   evaluate_mini_batch,
                   train_full_batch,
                   train_mini_batch,
                   tab_printer)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ACM")
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--r', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--metrics', type=str, nargs='+',
                    default=["micro-f1", "macro-f1"])
parser.add_argument("--monitor", type=str, default="metric")
parser.add_argument('--mini_batch', action='store_true')
parser.add_argument('--mask_lp', action='store_true')
parser.add_argument('--p', type=float, default=0.7)
parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 10])
parser.add_argument('--batch_size', type=int, default=1024)

try:
    args = parser.parse_args()
    table = tab_printer(args)
except:
    parser.print_help()
    exit(0)

root = 'data/'

seed_everything(args.seed)
seed = args.seed
dataset = args.dataset
hidden = args.hidden
lr = args.lr
epochs = args.epochs
dropout = args.dropout
r = args.r
alpha = args.alpha
beta = args.beta
mask_lp = args.mask_lp
p = args.p
project = False if dataset in ['ACM', 'IMDB'] else True
mini_batch = args.mini_batch
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available(
) and args.device >= 0 else 'cpu')
logger = setup_logger(output=f'log.txt')
logger.info(f'\n{table}')

############################
metrics = args.metrics
for metric in metrics:
    assert metric in ['acc', 'micro-f1', 'macro-f1', 'ap', 'auc']
monitor = args.monitor
assert monitor in ['loss', 'metric']
############################

############################
batch_size = args.batch_size
num_neighbors = args.num_neighbors
############################

if dataset == 'ACM':
    data = HGBDataset(root=f'{root}/HGBDataset', name='acm')[0]
    data['term'].x = torch.zeros(data['term'].num_nodes, 1)
    metapaths = [
                [('author', 'paper'), ('paper', 'author')],
                [('author', 'paper'), ('paper', 'term'),
                 ('term', 'paper'), ('paper', 'author')],
    ]
    # metapaths_to_add = [
    #                     [('paper', 'author'), ('author', 'paper')],
    #                     # [('paper', 'subject'), ('subject', 'paper')],
    #                     # [('paper', 'term'), ('term', 'paper')]
    # ]
    # data = T.AddMetaPaths(metapaths_to_add, drop_orig_edge_types=False)(copy(data))
elif dataset == 'DBLP':
    data = HGBDataset(root=f'{root}/HGBDataset', name='dblp')[0]
    data['venue'].x = torch.eye(data['venue'].num_nodes)
    data['term'].x = torch.eye(data['term'].num_nodes)
    data['paper'].x = torch.eye(data['paper'].num_nodes)
    metapaths = [
                [('paper', 'author'), ('author', 'paper')],
                [('author', 'paper'), ('paper', 'venue'),
                 ('venue', 'paper'), ('paper', 'author')],
    ]
elif dataset == 'IMDB':
    data = HGBDataset(root=f'{root}/HGBDataset', name='imdb')[0]
    data['keyword'].x = torch.zeros(data['keyword'].num_nodes, 1)
    metapaths = [
                [('director', 'movie'), ('movie', 'actor')],
                [('movie', 'keyword'), ('keyword', 'movie')],
    ]
    # metapaths_to_add = [[('movie', 'director'), ('director', 'movie')],
    #                     [('movie', 'actor'), ('actor', 'movie')],
    #                     [('movie', 'keyword'), ('keyword', 'movie')],
    #                    ]
    # data = T.AddMetaPaths(metapaths_to_add, drop_orig_edge_types=False)(copy(data))

elif dataset == 'FreeBase':
    data = HGBDataset(root=f'{root}/HGBDataset', name='freebase')[0]
    for nt in data.node_types:
        if data[nt].get('x') is None:
            data[nt].x = torch.eye(data[nt].num_nodes)
    metapaths = None
elif dataset == 'MAG':
    data = OGB_MAG(root, preprocess='metapath2vec',
                   transform=T.ToUndirected())[0]
    metapaths = [
                [('paper', 'author'), ('author', 'institution'),
                 ('institution', 'author'), ('author', 'paper')],
                [('author', 'paper'), ('paper', 'field_of_study'),
                 ('field_of_study', 'paper'), ('paper', 'author')],
    ]
elif dataset == 'RCDD':
    data = RCDDataset(f'{root}/RCDDataset')[0]
    metapaths = [
                [('f', 'item'), ('item', 'b')],
                [('b', 'item'), ('item', 'f')],
    ]

#     metapaths_to_add = [[('item', 'f'), ('f', 'item')], [('item', 'b'), ('b', 'item')]]
#     data = T.AddRandomMetaPaths(metapaths_to_add, drop_orig_edge_types=False)(copy(data))

else:
    raise ValueError(dataset)

logger.info(data)
logger.info("=" * 70)
logger.info("Node feature statistics")
for name, value in data.x_dict.items():
    logger.info(f"Name: {name}, feature shape: {value.size()}")

logger.info("=" * 70)

logger.info("Edge statistics")
for name, value in data.edge_index_dict.items():
    logger.info(f"Relation: {name}, edge shape: {value.size()}")
logger.info("=" * 70)

for et in data.edge_types:
    data[et].edge_index = sort_edge_index(data[et].edge_index)

node_types = [t for t in data.node_types if data[t].get('y') is not None]
assert len(node_types) == 1
node_type = node_types[0]
logger.info(f'Node type for classification: {node_type}')

if dataset == 'FreeBase':
    data = data.to(device, 'y')  # 预先放入显存
else:
    data = data.to(device, 'x', 'y')  # 预先放入显存

if data[node_type].y.squeeze().ndim == 1:
    num_classes = data[node_type].y.max().item() + 1
else:
    num_classes = data[node_type].y.size(-1)


train_mask = data[node_type].train_mask
test_mask = data[node_type].test_mask
val_mask = data[node_type].get('val_mask')
if val_mask is None:
    train_idx = train_mask.nonzero().view(-1)
    num_valid_samples = int(train_idx.size(0)*0.2)
    valid_idx = train_idx[torch.randperm(train_idx.size(0))[
        :num_valid_samples]]
    train_mask[valid_idx] = False
    val_mask = index_to_mask(valid_idx, data[node_type].num_nodes)


train_ratio = train_mask.float().mean().item()
test_ratio = test_mask.float().mean().item()
val_ratio = val_mask.float().mean().item() if val_mask is not None else 0

logger.info(
    f'Train ratio: {train_ratio:.2%}, Valid ratio: {val_ratio:.2%}, Test ratio: {test_ratio:.2%}')

train_data = copy(data)
test_data = copy(data)

if mask_lp:
    logger.info("Add labels for label propagation...")
    y = train_data[node_type].y.clone()
    if y.ndim > 1:
        y = y.long()
        row, col = y.nonzero().T
        y[y == 0] = num_classes
        y[row, col] = col
    y[y == -1] = num_classes  # mask unknown nodes
    y[test_mask] = num_classes  # mask test nodes
    test_data[node_type].y_emb = y.clone()

    if val_mask is not None:
        y[val_mask] = num_classes  # mask validation nodes
    train_data[node_type].y_emb = y.clone()
    del y

if mini_batch:
    train_loader = NeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(node_type, train_mask),
        num_workers=6,
        shuffle=True,
    )

    test_loader = NeighborLoader(
        test_data,
        num_neighbors=[-1, -1] if dataset in ['ACM',
                                              'DBLP', 'IMDB'] else num_neighbors,
        batch_size=batch_size*2,
        input_nodes=(node_type, test_mask),
        num_workers=6,
        shuffle=False,
    )

    if val_mask is not None:
        val_loader = NeighborLoader(
            train_data,
            num_neighbors=[-1, -1] if dataset in ['ACM',
                                                  'DBLP', 'IMDB'] else num_neighbors,
            batch_size=batch_size*2,
            input_nodes=(node_type, val_mask),
            num_workers=6,
            shuffle=False,
        )

# for seed in range(5):
model = HeteroGNN(data.metadata(),
                  in_channels=data[node_type].x.size(1),
                  hidden_channels=hidden,
                  out_channels=num_classes,
                  dropout=dropout,
                  project=project).to(device)
# logger.info(model)

model.node_type = node_type
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_metrics = None
best_val_loss = 1e5
best_val_metric = 0

for epoch in range(1, epochs+1):

    if mini_batch:
        loss = train_mini_batch(model, optimizer, train_loader, device,
                                p=p,
                                metapaths=metapaths, r=r,
                                alpha=alpha, beta=beta)
        test_loss, test_metrics = evaluate_mini_batch(
            model, test_loader, device, metrics=metrics)
        if val_mask is not None:
            val_loss, val_metrics = evaluate_mini_batch(
                model, val_loader, device, metrics=metrics)
        else:
            val_loss, val_metrics = test_loss, test_metrics
    else:
        loss = train_full_batch(model, optimizer, train_data, train_mask, device,
                                p=p,
                                metapaths=metapaths, r=r,
                                alpha=alpha, beta=beta)
        test_loss, test_metrics = evaluate_full_batch(
            model, test_data, test_mask, device, metrics=metrics)
        if val_mask is not None:
            val_loss, val_metrics = evaluate_full_batch(
                model, train_data, val_mask, device, metrics=metrics)
        else:
            val_loss, val_metrics = test_loss, test_metrics

    if monitor == 'loss':
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = test_metrics
    else:
        if val_metrics[0] > best_val_metric:
            best_val_metric = val_metrics[0]
            best_metrics = test_metrics

    logger.info(f'Epoch: {epoch}, Loss: {loss:.4f}')
    for metric, val_metric, test_metric, best_metric in zip(metrics, val_metrics, test_metrics, best_metrics):
        logger.info(
            f'Valid-{metric}: {val_metric:.2%}, Test-{metric}: {test_metric:.2%}, Best-{metric}: {best_metric:.2%}')

for metric, best_metric in zip(metrics, best_metrics):
    logger.info(f'Final-{metric}: {best_metric:.2%}')
