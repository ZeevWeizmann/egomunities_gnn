"""Inductive training and evaluation of PPGN, FGNN, and GCN models on cryptocurrency transaction graphs for illicit address detection."""


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from data_loader import load_graph_and_labels, load_address_embeddings
from egonets import extract_k_hop_illicit_clusters
from model import NodePPGN_3WL, FolkloreGNN, NodeGCN, subgraph_to_tensor_ppgn
from alignment import compute_penalty

# ===================== Reproducibility =====================
SEED = 42
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Load graph =====================
df, merged, illicit_set, addresses = load_graph_and_labels()
known_ids = set(addresses['address_id'])
df = df[df['addr_id1'].isin(known_ids) & df['addr_id2'].isin(known_ids)]

# ===================== Embeddings =====================
USE_EMBEDDINGS = False  # Set to True if address embeddings are available
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

# ===================== Ego-subgraphs =====================
subgraphs, _ = extract_k_hop_illicit_clusters(df, illicit_set, k=1)
egonet_nodes = set().union(*[G.nodes() for G in subgraphs])
print(f"[Egonets] Unique nodes: {len(egonet_nodes)}")

def convert_subgraphs_to_tensors(graph_list):
    return [
        subgraph_to_tensor_ppgn(
            G, illicit_set,
            device=device,
            emb_matrix=address_embeddings if has_embeddings else None,
            node_id_to_address=node_id_to_address if has_embeddings else None
        )
        for G in graph_list
    ]
all_tensors = convert_subgraphs_to_tensors(subgraphs)

# ===================== Split (egonet) =====================
nodes = sorted(list(egonet_nodes))
id2idx = {nid: i for i, nid in enumerate(nodes)}
y_global = np.array([1 if nid in illicit_set else 0 for nid in nodes])

idx_all = np.arange(len(nodes))
train_nodes, temp_nodes = train_test_split(
    idx_all, test_size=0.30, stratify=y_global, random_state=SEED
)
val_nodes, test_nodes = train_test_split(
    temp_nodes, test_size=0.50, stratify=y_global[temp_nodes], random_state=SEED
)
train_node_ids = {nodes[i] for i in train_nodes}
val_node_ids   = {nodes[i] for i in val_nodes}
test_node_ids  = {nodes[i] for i in test_nodes}

# ===================== Split (global graph) =====================
all_nodes = sorted(known_ids)
y_all_global = np.array([1 if nid in illicit_set else 0 for nid in all_nodes])
idx_all_global = np.arange(len(all_nodes))

train_nodes_global, temp_nodes_global = train_test_split(
    idx_all_global, test_size=0.30, stratify=y_all_global, random_state=SEED
)
val_nodes_global, test_nodes_global = train_test_split(
    temp_nodes_global, test_size=0.50, stratify=y_all_global[temp_nodes_global], random_state=SEED
)
train_node_ids_global = {all_nodes[i] for i in train_nodes_global}
val_node_ids_global   = {all_nodes[i] for i in val_nodes_global}
test_node_ids_global  = {all_nodes[i] for i in test_nodes_global}

print(f"[Split Ego] train={len(train_nodes)}, val={len(val_nodes)}, test={len(test_nodes)}")
print(f"[Split Global] train={len(train_nodes_global)}, val={len(val_nodes_global)}, test={len(test_nodes_global)}")

# ===================== Helpers =====================
def eval_on_indices(model, tensors, indices, split_nodes, gcn_mode=False, data=None, is_global=False, align_with_test=False):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        if gcn_mode:
            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            node_ids = all_nodes if is_global else nodes
            split_ids = test_node_ids_global if is_global else split_nodes

            for i, nid in enumerate(node_ids):
                if nid in split_ids:
                    all_true.append(np.array([int(data.y[i].cpu())]))
                    all_pred.append(np.array([int(preds[i].cpu())]))

            if align_with_test:
                z_all = model.get_embeddings(data.x, data.edge_index)
                z_train = z_all[[nid in (train_node_ids_global if is_global else train_node_ids) for nid in node_ids]]
                z_test  = z_all[[nid in (test_node_ids_global if is_global else test_node_ids) for nid in node_ids]]
                if z_train.size(0) > 0 and z_test.size(0) > 0:
                    penalty, err = compute_penalty(F.normalize(z_train, dim=1), F.normalize(z_test, dim=1), beta=50.0)
                    print(f"[Eval penalty] {model.__class__.__name__} train↔test penalty={penalty.item():.4f}, err={err.mean().item():.4f}")

        else:
            for gi in indices:
                X, Y, mask, id_map = tensors[gi]
                logits = model(X)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                node_ids = np.array(list(id_map.values()))
                y_all = Y[mask].cpu().numpy()
                p_all = preds[mask].cpu().numpy()

                split_mask = np.array([nid in split_nodes for nid in node_ids])
                y_true = y_all[split_mask]
                y_pred = p_all[split_mask]

                if len(y_true) > 0:
                    all_true.append(y_true)
                    all_pred.append(y_pred)

                if align_with_test:
                    z_all = model.get_embeddings(X)
                    z_train = z_all[mask][[nid in train_node_ids for nid in node_ids]]
                    z_test  = z_all[mask][[nid in test_node_ids for nid in node_ids]]
                    if len(z_train) > 0 and len(z_test) > 0:
                        penalty, err = compute_penalty(F.normalize(z_train, dim=1), F.normalize(z_test, dim=1), beta=50.0)
                        print(f"[Eval penalty] {model.__class__.__name__} train↔test penalty={penalty.item():.4f}, err={err.mean().item():.4f}")

    if len(all_true) == 0:
        return 0.0, np.zeros((2,2)), [], []
    yT = np.concatenate(all_true)
    yP = np.concatenate(all_pred)
    return f1_score(yT, yP, average="macro", zero_division=0), confusion_matrix(yT,yP,labels=[0,1]), yT, yP

def save_reports(name, y_true, y_pred, cm):
    rep = classification_report(y_true,y_pred,labels=[0,1],
                                target_names=["non-illicit","illicit"],
                                zero_division=0,output_dict=True)
    pd.DataFrame(rep).transpose().to_excel(f"{name}_classification_report.xlsx")
    pd.DataFrame(cm,
                 index=["true_non-illicit","true_illicit"],
                 columns=["pred_non-illicit","pred_illicit"]).to_excel(f"{name}_confusion_matrix.xlsx")
    return rep

# ===================== Training =====================
def train_model(model, name, tensors=None, data=None, gcn_mode=False, is_global=False, beta=50.0):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state, best_val, patience, ctr = None, -1, 10, 0

    for epoch in range(1, 201):
        model.train()
        opt.zero_grad()

        if gcn_mode:
            logits = model(data.x, data.edge_index)
            node_ids = all_nodes if is_global else nodes
            train_ids = train_node_ids_global if is_global else train_node_ids
            val_ids   = val_node_ids_global if is_global else val_node_ids
            test_ids  = test_node_ids_global if is_global else test_node_ids

            train_mask = torch.tensor([nid in train_ids for nid in node_ids], device=device)
            val_mask   = torch.tensor([nid in val_ids for nid in node_ids], device=device)
            test_mask  = torch.tensor([nid in test_ids for nid in node_ids], device=device)

            loss_cls = loss_fn(logits[train_mask], data.y[train_mask])

            z_all = model.get_embeddings(data.x, data.edge_index)
            z_train, z_val, z_test = z_all[train_mask], z_all[val_mask], z_all[test_mask]

            penalty_val, err_val = compute_penalty(F.normalize(z_train, dim=1), F.normalize(z_val, dim=1), beta=beta) if z_val.size(0)>0 else (torch.tensor(0.,device=device),torch.tensor(0.,device=device))
            penalty_test, err_test = compute_penalty(F.normalize(z_train, dim=1), F.normalize(z_test, dim=1), beta=beta) if z_test.size(0)>0 else (torch.tensor(0.,device=device),torch.tensor(0.,device=device))

            loss = loss_cls + penalty_val + penalty_test
            print(f"[Train {name}] Epoch {epoch} Loss_cls={loss_cls.item():.4f}, Pen_val={penalty_val.item():.4f}, Pen_test={penalty_test.item():.4f}")

        else:
            # PPGN/FGNN training loop
            loss_cls, loss_penalty = 0, 0
            for X, Y, mask, id_map in tensors:
                logits = model(X)
                node_ids = np.array(list(id_map.values()))
                train_mask_local = np.array([nid in train_node_ids for nid in node_ids])
                val_mask_local   = np.array([nid in val_node_ids for nid in node_ids])
                test_mask_local  = np.array([nid in test_node_ids for nid in node_ids])

                if train_mask_local.sum() == 0: continue
                loss_cls += loss_fn(logits[mask][train_mask_local], Y[mask][train_mask_local].float())

                z_all = model.get_embeddings(X)[mask]
                z_train, z_val, z_test = z_all[train_mask_local], z_all[val_mask_local], z_all[test_mask_local]

                if len(z_train)>0 and len(z_val)>0:
                    lp_val, _ = compute_penalty(F.normalize(z_train, dim=1), F.normalize(z_val, dim=1), beta=beta)
                    loss_penalty += lp_val
                if len(z_train)>0 and len(z_test)>0:
                    lp_test, _ = compute_penalty(F.normalize(z_train, dim=1), F.normalize(z_test, dim=1), beta=beta)
                    loss_penalty += lp_test

            loss = loss_cls + loss_penalty
            print(f"[Train {name}] Epoch {epoch} Loss_cls={loss_cls.item():.4f}, Penalty={loss_penalty.item():.4f}")

        loss.backward(); opt.step()

        val_f1, _, _, _ = eval_on_indices(model, tensors, range(len(tensors)) if not gcn_mode else None,
                                          val_node_ids_global if (gcn_mode and is_global) else val_node_ids,
                                          gcn_mode, data, is_global)
        if val_f1 > best_val:
            best_val, best_state, ctr = val_f1, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            ctr += 1
        if ctr >= patience: break

    if best_state:
        model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    return model

# ===================== Wrappers =====================
def run_ppgn(): return train_and_eval("ppgn", NodePPGN_3WL(in_channels=in_channels, hidden_dim=16).to(device))
def run_fgnn(): return train_and_eval("fgnn", FolkloreGNN(in_channels=in_channels, hidden_dim=16).to(device))
def run_gcn_egonets():
    df_ego = df[df['addr_id1'].isin(egonet_nodes) & df['addr_id2'].isin(egonet_nodes)]
    edge_index = [[id2idx[row['addr_id1']], id2idx[row['addr_id2']]] for _, row in df_ego.iterrows()]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.zeros((len(nodes), embedding_dim))
    for nid, idx in id2idx.items():
        if nid in address_embeddings: x[idx] = torch.tensor(address_embeddings[nid])
    y = torch.tensor([1 if nid in illicit_set else 0 for nid in nodes], dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    return train_and_eval("gcn_egonets", NodeGCN(x.size(1), 32).to(device), gcn_mode=True, data=data)

def run_gcn_global():
    id2idx_global = {nid:i for i,nid in enumerate(all_nodes)}
    edge_index = [[id2idx_global[row['addr_id1']],id2idx_global[row['addr_id2']]] for _,row in df.iterrows()]
    edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous()
    x=torch.zeros((len(all_nodes),embedding_dim))
    for nid,idx in id2idx_global.items():
        if nid in address_embeddings: x[idx]=torch.tensor(address_embeddings[nid])
    y=torch.tensor([1 if nid in illicit_set else 0 for nid in all_nodes],dtype=torch.float32)
    data=Data(x=x,edge_index=edge_index,y=y).to(device)
    return train_and_eval("gcn_global", NodeGCN(x.size(1),32).to(device), gcn_mode=True, data=data, is_global=True)

def train_and_eval(name, model, gcn_mode=False, data=None, is_global=False):
    model = train_model(model, name, tensors=all_tensors, data=data, gcn_mode=gcn_mode, is_global=is_global, beta=50.0)
    test_f1, cm, yT, yP = eval_on_indices(model, all_tensors, range(len(all_tensors)) if not gcn_mode else None,
                                          test_node_ids_global if (gcn_mode and is_global) else test_node_ids,
                                          gcn_mode, data, is_global, align_with_test=True)
    rep = save_reports(name, yT, yP, cm)
    return rep

# ===================== Run all and compare =====================
# rep_gcn_glob = run_gcn_global()
# run rep_gcn_glob if you have enough memory (anyway it is not competitive)
rep_ppgn = run_ppgn()
rep_fgnn = run_fgnn()
rep_gcn_ego = run_gcn_egonets()

df_compare=pd.DataFrame({
    "Metric":["Precision","Recall","F1"],
    # "GCN_global_illicit":[rep_gcn_glob["illicit"]["precision"],rep_gcn_glob["illicit"]["recall"],rep_gcn_glob["illicit"]["f1-score"]],
    "PPGN_illicit":[rep_ppgn["illicit"]["precision"],rep_ppgn["illicit"]["recall"],rep_ppgn["illicit"]["f1-score"]],
    "FGNN_illicit":[rep_fgnn["illicit"]["precision"],rep_fgnn["illicit"]["recall"],rep_fgnn["illicit"]["f1-score"]],
    "GCN_egonets_illicit":[rep_gcn_ego["illicit"]["precision"],rep_gcn_ego["illicit"]["recall"],rep_gcn_ego["illicit"]["f1-score"]],
})
print("\n=== Final model comparison ===")
print(df_compare)
df_compare.to_excel("comparison_all_models.xlsx",index=False)
print("Saved comparison_all_models.xlsx")
