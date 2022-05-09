import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import string
import re
from itertools import *
import sklearn
from sklearn import linear_model
import sklearn.metrics as Metric
import csv
import argparse
import torch.optim as optim
import time
from utils import  EarlyStopping
from sklearn.metrics import f1_score
import scipy.sparse as sp
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
from dgl.nn.pytorch import GATConv
class features(nn.Module):
    def __init__(self,in_size,hidden_size):
        super(features, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 384, bias=False))
    def forward(self, adj,z):
        adj=normalize(adj)
        adj=torch.from_numpy(adj)
        adj=adj.float()
        z=z.float()
        z=torch.mm(adj,z)
        w = self.project(z)
        return w

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)



class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.layer1=features(384,64)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Sequential(nn.Linear(hidden_size * num_heads[-1], out_size), nn.Sigmoid())

    def forward(self, g,adj,h):
        h=torch.mm(adj,h)
        h=h.float()
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)