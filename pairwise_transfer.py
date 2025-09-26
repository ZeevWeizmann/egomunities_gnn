"""
This script performs pairwise transfer learning experiments on a set of k-hop subgraphs
extracted from a larger graph. It supports three types of GNN models: PPGN, GCN, and FolkloreGNN.
It includes data loading, preprocessing, model training with alignment penalties, and evaluation.
"""
# === Standard imports ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report


# === Imports from  project ===
from alignment import compute_penalty
from data_loader import (
    load_graph_and_labels,
    load_address_embeddings,
    build_pyg_graphs_from_subgraphs,
)
from egonets import extract_k_hop_illicit_clusters, compute_triangle_stats
from model import NodePPGN_3WL, NodeGCN, FolkloreGNN, subgraph_to_tensor_ppgn

# --- Reproducibility ---
SEED = 42
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
set_seed(SEED)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load graph and address data ---
print("Loading and preprocessing graph data...")
df, merged, illicit_set, addresses = load_graph_and_labels()
df_filtered = df
known_ids = set(addresses["address_id"])
df_filtered = df_filtered[
    df_filtered["addr_id1"].isin(known_ids) &
    df_filtered["addr_id2"].isin(known_ids)
]

# --- Load address embeddings ---
print("Loading address embeddings...")
USE_EMBEDDINGS = False # Set to True if you want to use address embeddings
if USE_EMBEDDINGS:
    address_embeddings = load_address_embeddings()
else:
    address_embeddings = {}

has_embeddings = bool(address_embeddings) and len(address_embeddings) > 0
embedding_dim = next(iter(address_embeddings.values())).shape[0] if has_embeddings else 0
if not has_embeddings:
    degree_map = df.groupby('addr_id1').size().add(df.groupby('addr_id2').size(), fill_value=0)
    degree_map = degree_map.to_dict()
    def get_degree_features(nid):
        deg=degree_map.get(nid, 0)
        illicit_flag=1 if nid in illicit_set else 0
        return torch.tensor([deg, illicit_flag], dtype=torch.float32)
    
    address_embeddings = {nid: get_degree_features(nid) for nid in known_ids}
    embedding_dim = 2
    in_channels = 2
else:
    in_channels = 2 + 2 * embedding_dim

node_id_to_address = dict(zip(addresses["address_id"], addresses["address"])) if has_embeddings else None

# --- Extract k-hop subgraphs ---
print("Extracting k-hop subgraphs...")
subgraphs, sizes = extract_k_hop_illicit_clusters(df_filtered, illicit_set, k=1)

# --- Convert subgraphs to tensors (for PPGN/FGNN) ---
def convert_subgraphs_to_tensors(graph_list):
    return [
        subgraph_to_tensor_ppgn(
            G,
            illicit_set,
            device=device,
            emb_matrix=address_embeddings if has_embeddings else None,
            node_id_to_address=node_id_to_address if has_embeddings else None,
        )
        for G in graph_list
    ]

# --- Utility: pos_weight ---
def compute_pos_weight(y: torch.Tensor, mask: torch.Tensor, device):
    y_masked = y[mask]
    n_pos = int((y_masked == 1).sum().item())
    n_neg = int((y_masked == 0).sum().item())
    if n_pos > 0:
        return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    else:
        return torch.tensor([1.0], dtype=torch.float32, device=device)

# --- Utility: get masks for PyG ---
def _get_masks(batch):
    if hasattr(batch, "train_mask"):
        return batch.train_mask
    elif hasattr(batch, "mask"):
        return batch.mask
    else:
        return torch.ones(batch.num_nodes, dtype=torch.bool, device=batch.x.device)

# --- Build PyG graphs for GCN ---
print("Converting to PyG graphs...")
pyg_graphs = build_pyg_graphs_from_subgraphs(
    subgraphs,
    illicit_set,
    address_embeddings=address_embeddings,
    node_id_to_address=node_id_to_address,
)

# --- Unified runner ---
from sklearn.metrics import classification_report, confusion_matrix

# --- Unified runner ---
def run_pair(model_name, train_idx, test_idx, hidden_dim=16, lr=0.01, epochs=100, beta=100.0):
    print(f"\n====== {model_name.upper()} transfer: {train_idx} → {test_idx} ======")
    set_seed(SEED)

    # Common: get test triangle stats
    n_tri, tri_density = compute_triangle_stats(subgraphs[test_idx])

    if model_name.lower() == "ppgn":
        X_A, Y_A, mask_A, _ = convert_subgraphs_to_tensors([subgraphs[train_idx]])[0]
        X_B, Y_B, mask_B, _ = convert_subgraphs_to_tensors([subgraphs[test_idx]])[0]

        model = NodePPGN_3WL(in_channels=in_channels, hidden_dim=hidden_dim).to(device)
        pos_weight = compute_pos_weight(Y_A, mask_A, device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            model.train()
            logits_A = model(X_A)
            loss_cls = loss_fn(logits_A[mask_A], Y_A[mask_A])

            z_A = F.normalize(model.get_embeddings(X_A), dim=1)
            with torch.no_grad():
                z_B = F.normalize(model.get_embeddings(X_B), dim=1)
            penalty, _ = compute_penalty(z_A, z_B, beta=beta)

            loss = loss_cls + penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_B = model(X_B)
            probs_B = torch.sigmoid(logits_B)
            preds_B = (probs_B > 0.5).float()

        y_true, y_pred = Y_B[mask_B].cpu().numpy(), preds_B[mask_B].cpu().numpy()

    elif model_name.lower() == "gcn":
        train_graph = pyg_graphs[train_idx]
        test_graph = pyg_graphs[test_idx]
        train_loader = DataLoader([train_graph], batch_size=1, shuffle=False)
        test_loader = DataLoader([test_graph], batch_size=1, shuffle=False)

        in_channels_gcn = train_graph.x.shape[1]
        model = NodeGCN(input_dim=in_channels_gcn, hidden_dim=hidden_dim).to(device)
        mask_A = _get_masks(train_graph)
        pos_weight = compute_pos_weight(train_graph.y, mask_A, device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                logits_A = model(batch.x, batch.edge_index)
                mask_A = _get_masks(batch)
                Y_A = batch.y.float()
                loss_cls = loss_fn(logits_A[mask_A], Y_A[mask_A])

                z_A = model.get_embeddings(batch.x, batch.edge_index)
                with torch.no_grad():
                    batch_B = next(iter(test_loader)).to(device)
                    z_B = model.get_embeddings(batch_B.x, batch_B.edge_index)
                penalty, _ = compute_penalty(F.normalize(z_A, dim=1), F.normalize(z_B, dim=1), beta=beta)

                loss = loss_cls + penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            batch = test_graph.to(device)
            logits = model(batch.x, batch.edge_index)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

        mask_B = _get_masks(batch)
        y_true, y_pred = batch.y[mask_B].cpu().numpy(), preds[mask_B].cpu().numpy()

    elif model_name.lower() == "fgnn":
        X_A, Y_A, mask_A, _ = convert_subgraphs_to_tensors([subgraphs[train_idx]])[0]
        X_B, Y_B, mask_B, _ = convert_subgraphs_to_tensors([subgraphs[test_idx]])[0]

        model = FolkloreGNN(in_channels=in_channels, hidden_dim=hidden_dim).to(device)
        pos_weight = compute_pos_weight(Y_A, mask_A, device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            model.train()
            logits_A = model(X_A)
            loss_cls = loss_fn(logits_A[mask_A], Y_A[mask_A])

            z_A = F.normalize(model.get_embeddings(X_A), dim=1)
            with torch.no_grad():
                z_B = F.normalize(model.get_embeddings(X_B), dim=1)
            penalty, _ = compute_penalty(z_A, z_B, beta=beta)

            loss = loss_cls + penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_B = model(X_B)
            probs_B = torch.sigmoid(logits_B)
            preds_B = (probs_B > 0.5).float()

        y_true, y_pred = Y_B[mask_B].cpu().numpy(), preds_B[mask_B].cpu().numpy()

    # === Evaluation ===
    n_true_illicit = (y_true == 1).sum()
    n_nodes = len(subgraphs[test_idx].nodes)
    if n_true_illicit == 0:
        return {
            "model": model_name,
            "train_graph": train_idx,
            "test_graph": test_idx,
            "precision_illicit": 0.0,
            "recall_illicit": 0.0,
            "f1_illicit": 0.0,
            "support_illicit": 0,
            "TP": 0, "FP": 0, "FN": 0, "TN": 0,
            "n_triangles": n_tri,
            "tri_density": tri_density,
        }

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["non-illicit", "illicit"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "model": model_name,
        "train_graph": train_idx,
        "test_graph": test_idx,
        "precision_illicit": report["illicit"]["precision"],
        "recall_illicit": report["illicit"]["recall"],
        "f1_illicit": report["illicit"]["f1-score"],
        "support_illicit": report["illicit"]["support"],
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "n_triangles": n_tri,
        "tri_density": tri_density,
        "n_nodes": n_nodes, 
    }


# --- Define pairs ---

extra_test_nodes =  list(range(0, 1066))

pairs = [(50, t) for t in extra_test_nodes]
# --- Run all models ---
results = []
for model_name in ["ppgn", "gcn", "fgnn"]:
    for tr, te in pairs:
        try:
            res = run_pair(model_name, tr, te, hidden_dim=16, lr=0.01, epochs=100, beta=100.0)
            results.append(res)
        except Exception as e:
            print(f"[Error] {model_name} ({tr}→{te}): {e}")

df_results = pd.DataFrame(results)
df_results.to_excel("manual_pair_results_all_models.xlsx", index=False)
print("\n Saved results to manual_pair_results_all_models.xlsx")
