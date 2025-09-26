""" Data loading and preprocessing for Bitcoin graph analysis project.
This module provides functions to load the Bitcoin graph data, illicit addresses,
and address statistics, as well as to prepare the data for training and testing
with PyTorch Geometric."""

import pandas as pd
import pyarrow.parquet as pq
import os
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch, random, numpy as np
from collections import Counter
from torch_geometric.utils import from_networkx
from torch_geometric.utils import from_networkx
import networkx as nx



# === Global paths (adjust as needed) ===
GRAPH_PATH = '/Users/zeevweizmann/Downloads/hacker_subgraph_k1 (1).parquet'
ILLICIT_CSV_PATH = '/Users/zeevweizmann/Downloads/crypto_classification_results_illicit.csv'
ADDRESS_STATS_PATH = '/Users/zeevweizmann/projects/Internship/btc_graph/address_stats.parquet'
EMBEDDINGS_DIR = '/Users/zeevweizmann/Downloads/bitcoin_embeddings/max_pool'

# === Data loading and preprocessing functions for the Bitcoin graph analysis project ===
def load_graph_and_labels():
    """
    Load the graph data and illicit addresses, merging them with address statistics.
    Returns:
        df (pd.DataFrame): DataFrame containing the graph data.
        merged (pd.DataFrame): DataFrame containing illicit addresses merged with address statistics.
        illicit_set (set): Set of illicit address IDs.
        addresses (pd.DataFrame): DataFrame containing address statistics.
    """
    table = pq.read_table(GRAPH_PATH)
    df = table.to_pandas()
    illicit_addresses = pd.read_csv(ILLICIT_CSV_PATH)
    illicit_addresses = illicit_addresses.drop_duplicates(subset='address', keep='first')
    addresses = pd.read_parquet(ADDRESS_STATS_PATH)
    merged = pd.merge(illicit_addresses, addresses, on='address', how='inner')
    illicit_set = set(merged['address_id'].astype(np.int64))
    return df, merged,illicit_set, addresses
def load_address_embeddings():
    """ 
    Load address embeddings from a directory containing .pt files.
    Each file is named after the address (without the .pt extension) and contains a tensor.
    The embeddings are assumed to be stored in a directory structure like:
    /path/to/embeddings/
        address1.pt
        address2.pt
        ...
    Returns:
        dict: A dictionary mapping addresses to their embeddings.
    """
    emb_dict = {}
    for fname in os.listdir(EMBEDDINGS_DIR):
        if fname.endswith(".pt"):
            address = fname.replace(".pt", "")
            emb = torch.load(os.path.join(EMBEDDINGS_DIR, fname))
            emb_dict[address] = emb
    return emb_dict

def build_pyg_graphs_from_subgraphs(subgraphs, illicit_set, address_embeddings=None, node_id_to_address=None):
    """
    Convert list of NetworkX subgraphs into PyG Data objects with node features and labels.
    
    Features:
        - if address_embeddings provided: use embeddings
        - otherwise: use node degree
    
    Labels:
        - 1 for illicit, 0 otherwise

    Returns:
        List[torch_geometric.data.Data]
    """
    

    pyg_graphs = []

    for sg in subgraphs:
        for n in sg.nodes():
            if address_embeddings and node_id_to_address:
                address = node_id_to_address.get(n)
                if address in address_embeddings:
                    features = address_embeddings[address].cpu().numpy()
                else:
                    features = np.zeros(next(iter(address_embeddings.values())).shape[0])
            else:
                # Fallback to degree
                features = [sg.degree[n]]

            sg.nodes[n]['x'] = features
            sg.nodes[n]['id'] = n

        data = from_networkx(sg)
        data.x = data.x.float()  # ensure float32
        data.y = torch.tensor([1 if n in illicit_set else 0 for n in sg.nodes()], dtype=torch.long)
        data.node_id = torch.tensor([sg.nodes[n]['id'] for n in sg.nodes()], dtype=torch.long)

        pyg_graphs.append(data)

    return pyg_graphs

#For GCN batch processing (if needed)
def prepare_data(pyg_graphs, batch_size=32, seed=42):
    """
    Prepare the data for training and testing by splitting the graphs into train and test sets,
    and creating DataLoaders for each set.
    Args:
        pyg_graphs (list): List of PyTorch Geometric graph objects.
        batch_size (int): Batch size for DataLoader.
        seed (int): Random seed for reproducibility.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    # Seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Split
    train_graphs, test_graphs = train_test_split(pyg_graphs, test_size=0.20, random_state=seed)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # Print class distribution
    train_labels = [label for batch in train_loader for label in batch.y.tolist()]
    test_labels = [label for batch in test_loader for label in batch.y.tolist()]

    print("Train class distribution:", Counter(train_labels))
    print("Test  class distribution :", Counter(test_labels))

    return train_loader, test_loader


# Check for duplicates by address
# """Check for duplicates in the illicit addresses CSV file by address.This is useful to ensure that each address is unique in the dataset.
# """
# df = pd.read_csv(ILLICIT_CSV_PATH)
# duplicates = df[df.duplicated('address', keep=False)].sort_values('address')
# print(f"Number of duplicates found by address: {len(duplicates)}")
# print(duplicates.head(10))



