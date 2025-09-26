"""This module contains various GNN models and utility functions for processing subgraphs.
It includes a PPGN model, a Folklore GNN, and a standard GCN model, as well as functions to convert subgraphs to tensors with embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def subgraph_to_tensor_ppgn(
    G,
    illicit_set,
    device='cpu',
    emb_matrix=None,
    node_id_to_address=None
):
    """
    Converts a graph to a tensor representation suitable for PPGN,
    without leaking labels into features.
    """
    node_ids = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_ids)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    n = len(node_ids)

    emb_dim = next(iter(emb_matrix.values())).shape[0] if emb_matrix else 0

    # adjacency + symmetry + embeddings
    d = 2 + 2 * emb_dim if emb_matrix else 2
    X = torch.zeros((n, n, d))

    # adjacency + symmetry
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        X[i, j, 0] = 1.0
        X[j, i, 0] = 1.0
        X[i, j, 1] = 1.0
        X[j, i, 1] = 1.0

    # embeddings
    if emb_matrix and node_id_to_address:
        for i, u in enumerate(node_ids):
            for j, v in enumerate(node_ids):
                addr_u = node_id_to_address.get(u)
                addr_v = node_id_to_address.get(v)
                if addr_u in emb_matrix:
                    X[i, j, 2:2+emb_dim] = emb_matrix[addr_u]
                if addr_v in emb_matrix:
                    X[i, j, 2+emb_dim:2+2*emb_dim] = emb_matrix[addr_v]

    # Labels 
    Y = torch.tensor(
        [1.0 if node in illicit_set else 0.0 for node in node_ids],
        dtype=torch.float32
    )
    mask = torch.ones(n, dtype=torch.bool)

    return X.to(device), Y.to(device), mask.to(device), idx_to_node
"""
# ------------------ PPGN Model (3-WL) ------------------
This model implements a PPGN (Parameterized Pairwise Graph Neural Network) based on the 3-WL algorithm.
It uses two linear transformations followed by element-wise multiplication and concatenation,
then computes a pairwise interaction matrix and aggregates it to produce node embeddings.
 NodePPGN_3WL is a PyTorch module that implements a PPGN model based on the 3-WL algorithm.
It uses two linear transformations to project the input features, computes pairwise interactions,
and aggregates the results to produce node embeddings and logits.
"""


class NodePPGN_3WL(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, hidden_dim)
        self.mlp2 = nn.Linear(in_channels, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, 1)

    def forward(self, X):  # X: [n, n, d]
        T1 = self.mlp1(X)
        T2 = self.mlp2(X)
        W = T1 * T2
        H = torch.cat([T1, W], dim=-1)
        Z = torch.einsum("ijh,kjh->ikh", H, H)
        H_final = Z.sum(dim=1)
        logits = self.classifier(H_final).squeeze(-1)
        return logits
    def get_embeddings(self, X):
        T1 = self.mlp1(X)
        T2 = self.mlp2(X)
        W = T1 * T2
        H = torch.cat([T1, W], dim=-1)
        Z = torch.einsum("ijh,kjh->ikh", H, H) # Z[i, k, h] aggregates pairwise interactions between H[i, j, :] and H[k, j, :] over shared index j
# This mimics the 3-WL update mechanism by computing similarity between neighborhoods  
        # Aggregate the pairwise interactions
        # by summing over the second dimension
        H_final = Z.sum(dim=1) 
        if H_final.dim() == 1:   # защита для n=1
            H_final = H_final.unsqueeze(0)
        # Apply ReLU activation and dropout
        H_final = F.relu(H_final)
        H_final = F.dropout(H_final, p=0.5, training=self.training)
        return H_final   

# ------------------ Folklore GNN ------------------
    """Folklore GNN" is a simple GNN model that projects input features into two separate spaces,
    computes pairwise interactions, and aggregates them to produce node embeddings and logits.
"""
class FolkloreGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.proj_i = nn.Linear(in_channels, hidden_dim)
        self.proj_j = nn.Linear(in_channels, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, X):  # X: [n, n, d]
        H_i = self.proj_i(X)
        H_j = self.proj_j(X)
        H_pair = H_i * H_j
        H_sum = H_pair.sum(dim=1)  # [n, hidden_dim]
        logits = self.classifier(H_sum).squeeze(-1)
        return logits
    def get_embeddings(self, x):
        h_i = self.proj_i(x)  # [n, n, hidden_dim]
        h_j = self.proj_j(x)
        h = h_i * h_j
        h = h.sum(dim=1)      # [n, hidden_dim]
        if h.dim() == 1:   # защита для n=1
            h = h.unsqueeze(0)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        return h 


# ------------------ Standard GCN (binary, BCE) ------------------
"""
This is a binary GCN (Graph Convolutional Network) model
that uses two GCNConv layers followed by a linear classifier
with a single logit per node.
It matches the setup of PPGN/FGNN (BCEWithLogitsLoss).
"""
class NodeGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)  # ⬅ один выход

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x).squeeze(-1)  # [n]

    def get_embeddings(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        return h

