
""" Utility functions for graph processing and analysis. """
import numpy as np
import networkx as nx


def filter_graph(df):
    """
    Filter the graph DataFrame to remove nodes with too many incoming or outgoing edges.
    Args:
        df (pd.DataFrame): DataFrame containing the graph data with columns 'addr_id1' and 'addr_id2'.
    Returns:
        pd.DataFrame: Filtered DataFrame with nodes that have fewer than 1000 incoming and outgoing edges.
    """
    in_degree = df['addr_id2'].value_counts()
    too_many_incoming = set(in_degree[in_degree > 1000].index)
    df_filtered_incoming = df[~df['addr_id2'].isin(too_many_incoming)]

    out_degree = df_filtered_incoming['addr_id1'].value_counts()
    too_many_outgoing = set(out_degree[out_degree > 1000].index)
    df_filtered = df_filtered_incoming[~df_filtered_incoming['addr_id1'].isin(too_many_outgoing)]

    return df_filtered

def build_graph(df_filtered):
    """
    Build a NetworkX graph from the filtered DataFrame.
    Args:
        df_filtered (pd.DataFrame): Filtered DataFrame with columns 'addr_id1' and 'addr_id2'.
    Returns:
        nx.Graph: A NetworkX graph constructed from the filtered DataFrame.
    """
    G = nx.from_pandas_edgelist(df_filtered, 'addr_id1', 'addr_id2')
    return G

def analyze_lost_illicit(G, merged):
    """
    Analyze which illicit addresses are not present in the graph.
    Args:
        G (nx.Graph): The NetworkX graph.
        merged (pd.DataFrame): DataFrame containing illicit addresses with 'address_id' column.
    Returns:
        tuple: A tuple containing:
            - lost_illicit (set): Set of illicit address IDs not present in the graph.
            - lost_df (pd.DataFrame): DataFrame of lost illicit addresses.
    """
    
    illicit_set = set(merged['address_id'].astype(np.int64))
    graph_nodes = set(G.nodes)
    lost_illicit = illicit_set - graph_nodes
    lost_df = merged[merged['address_id'].isin(lost_illicit)]
    return lost_illicit, lost_df

# --- Triangle statistics utilities ---
def count_triangles(G):
    """ Count the number of unique triangles in graph G (undirected).
    NetworkX counts triangles per node; each triangle is counted 3 times (once at each vertex).
    We divide the sum by 3 to get the number of unique triangles. """
    triangles_per_node = nx.triangles(G)
    total_triangles = sum(triangles_per_node.values()) // 3
    return total_triangles

def compute_tri_density(g):
    """ Compute triangle density (TriDensity) for subgraph g as in Friggeri et al. (2011):
        TriDensity(S) = Δ_in(S) / C(|S|, 3),
        where C(n,3) = n*(n-1)*(n-2)/6 is the maximum possible number of triangles.
        Returns a value in [0,1]. For n<3 returns 0.0. """
    n = g.number_of_nodes()
    if n < 3:
        return 0.0
    triangles = count_triangles(g)  # Δ_in(S)
    max_triangles = n * (n - 1) * (n - 2) / 6  # C(n, 3)
    if max_triangles == 0:
        return 0.0
    return triangles / max_triangles