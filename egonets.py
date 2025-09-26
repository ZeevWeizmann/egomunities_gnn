"""
Module for extracting and analyzing k-hop subgraphs (ego-nets) around illicit nodes in a graph.
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from utils import build_graph


def extract_k_hop_illicit_clusters(df_edges, illicit_set, k, return_ids=False, return_centers=False):
    """
    Extract k-hop subgraphs around illicit nodes from a DataFrame of edges.
    Each subgraph contains nodes within k hops of an illicit node.

    Parameters:
        df_edges (pd.DataFrame): DataFrame with columns 'addr_id1' and 'addr_id2' representing edges.
        illicit_set (set): Set of illicit address_ids.
        k (int): Number of hops to consider for the subgraph.
        return_ids (bool): If True, also return cluster IDs of illicit nodes.
        return_centers (bool): If True, also return the list of center nodes used to build each subgraph.

    Returns:
        list of nx.Graph: List of k-hop subgraphs around illicit nodes.
        list of int: Sizes of each subgraph (number of nodes).
        list (optional): Either cluster_ids or centers depending on return_ids / return_centers.
    """
    G = build_graph(df_edges)
    subgraphs = []
    used_nodes = set()
    cluster_ids = []
    centers = []  # Will store the center node for each subgraph
    illicit_in_graph = list(illicit_set & set(G.nodes()))  # Get only illicit nodes that are in the graph

    for illicit_node in illicit_in_graph:

        if illicit_node in used_nodes:
            continue  # Skip if this node is already included in a previous subgraph

        neighbors = nx.single_source_shortest_path_length(G, illicit_node, cutoff=k)


        neighborhood_nodes = set(neighbors.keys())  
        sg = G.subgraph(neighborhood_nodes).copy()  # Create subgraph
        subgraphs.append(sg)  # Save the subgraph
        cluster_ids.append(illicit_node)  # Track cluster ID
        centers.append(illicit_node)  # Track center node
        used_nodes.update(neighborhood_nodes)  # Mark these nodes as used

    sizes = [len(sg.nodes()) for sg in subgraphs]  # Get sizes of each subgraph

    # Return different formats based on flags
    if return_ids:
        return subgraphs, sizes, cluster_ids
    elif return_centers:
        return subgraphs, sizes, centers
    else:
        return subgraphs, sizes

    

def plot_khop_cluster_size_distribution(subgraphs):
    """
    Plot the distribution of k-hop subgraph sizes on a log-log scale.

    Parameters:
        subgraphs (list of nx.Graph): List of subgraphs
    Returns:
        None: Displays a plot of the size distribution.

    """
    sizes = [len(sg.nodes()) for sg in subgraphs] # Get sizes of each subgraph
    size_counts = Counter(sizes) # Count occurrences of each size
    sizes_sorted = sorted(size_counts.items()) # Sort sizes by size value
    x_vals, y_vals = zip(*sizes_sorted) # Unzip into x and y values 

    plt.figure(figsize=(8, 5))
    plt.loglog(x_vals, y_vals, marker='o', linestyle='None')
    plt.xlabel("k-hop neighborhood size (number of nodes)")
    plt.ylabel("Frequency")
    plt.title("Distribution of k-hop neighborhood sizes (log-log scale)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
def compute_cluster_illicit_stats(subgraphs, illicit_set, min_illicit=1, sort_by='illicit'):
    """
    Compute statistics for each subgraph: illicit and normal node counts.

    Parameters:
        subgraphs (list of nx.Graph): List of subgraphs
        illicit_set (set): Set of illicit address_ids
        min_illicit (int): Minimum number of illicit nodes to include in results
        sort_by (str): 'illicit', 'size', or 'ratio'

    Returns:
        List of tuples: (cluster_id, #illicit, #normal, total, illicit_ratio)
    """
    cluster_stats = []

    for i, sg in enumerate(subgraphs):
        nodes = set(sg.nodes()) # Get nodes in the subgraph
        num_illicit = sum(n in illicit_set for n in nodes) # Count illicit nodes
        num_normal = len(nodes) - num_illicit # Count normal nodes
        total = len(nodes)  # Total nodes in the subgraph
        ratio = num_illicit / total if total > 0 else 0.0
        cluster_stats.append((i, num_illicit, num_normal, total, ratio))

    # Filter clusters based on minimum illicit nodes
    filtered = [t for t in cluster_stats if t[1] >= min_illicit]

    #   Sort the filtered clusters based on the specified criterion
    if sort_by == 'illicit':
        filtered.sort(key=lambda x: x[1], reverse=True)
    elif sort_by == 'ratio':
        filtered.sort(key=lambda x: x[4], reverse=True)
    elif sort_by == 'size':
        filtered.sort(key=lambda x: x[3], reverse=True)

    return filtered

def compute_triangle_stats(G: nx.Graph):
    """
    Compute the number of unique triangles and triangle density in graph G.
    Parameters:
        G (nx.Graph): Input undirected graph
    Returns:
        tuple: (number of unique triangles, triangle density)
    """
    
    # nx.triangles возвращает словарь: {node: кол-во треугольников через этот узел}
    tri_count = sum(nx.triangles(G).values()) // 3  # делим на 3, т.к. каждый треугольник учитывается по вершинам

    n = G.number_of_nodes()
    if n < 3:
        tri_density = 0.0
    else:
        max_tri = n * (n - 1) * (n - 2) / 6  # C(n,3)
        tri_density = tri_count / max_tri if max_tri > 0 else 0.0

    return tri_count, tri_density