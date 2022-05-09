import networkx as nx
import numpy as np
import scipy
import pickle
import torch
import dgl
import torch
from sklearn.metrics import f1_score
import scipy.sparse as sp
import time
from utils import *
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans ,KMeans
from scipy import sparse
import scipy.io as sio
from model import HAN
import argparse
from utils import setup
from data_load import *
parser = argparse.ArgumentParser('HAN')
parse.add_argument("-d", "--dataset", help="dataset", default="dblp", type=str, required=False)
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('-ld', '--log-dir', type=str, default='results',
                    help='Dir for saving training results')
parser.add_argument('--hetero', action='store_true',
                    help='Use metapath coalescing with DGL\'s own dataset')
args = parser.parse_args().__dict__
args = setup(args)

dataset=args.dataset
if dataset=='dblp':
    g, adj, embeddings, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
        val_mask, test_mask=load_data_dblp()
else:
    g, adj, embeddings, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data_imdb()

model = HAN(num_meta_paths=len(g),
                in_size=384,
                hidden_size=args['hidden_units'],
                out_size=num_classes,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])
g = [graph.to(args['device']) for graph in g]
stopper = EarlyStopping(patience=args['patience'])
loss_fcn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                             weight_decay=args['weight_decay'])
t = time.time()
for epoch in range(args['num_epochs']):
    model.train()
    logits = model(g, adj,embeddings)

    loss = loss_fcn(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
    val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, adj, embeddings, labels, val_mask, loss_fcn)
    early_stop = stopper.step(val_loss.data.item(), val_acc, model)

    print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
          'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
        epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))
    if early_stop:
        break

stopper.load_checkpoint(model)
output = model(g, adj,embeddings)
test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, adj, embeddings, labels, test_mask, loss_fcn)
print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
    test_loss.item(), test_micro_f1, test_macro_f1))
print('time: {:.4f}s'.format(time.time() - t))







