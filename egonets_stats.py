"""
Compute statistics on k-hop illicit clusters extracted from a transaction graph.
This includes counting triangles and calculating triangle density for each cluster.
"""

from data_loader import load_graph_and_labels
from utils import filter_graph, build_graph,count_triangles, compute_tri_density
from egonets import extract_k_hop_illicit_clusters, plot_khop_cluster_size_distribution, compute_cluster_illicit_stats
import networkx as nx

# --- Load data ---
""" Load the graph and labels from the data loader module """
df, merged, illicit_set, addresses = load_graph_and_labels()  

# Print initial stats before filtering
print(f'Before filtering: {len(df)} edges, {len(df)} nodes, {len(illicit_set)} illicit addresses') 

known_ids = set(addresses['address_id'])  # Known address IDs from the addresses DataFrame

# === Step 1: Structural filter (optional) ===
# """ Example: filter the DataFrame to keep only edges with known address IDs """
# df_filtered_1 = filter_graph(df)
# G_filtered_1 = build_graph(df_filtered_1)
# illicit_in_filtered_1 = illicit_set & set(G_filtered_1.nodes)
# print(f"After structural filter: {G_filtered_1.number_of_nodes()} nodes, {G_filtered_1.number_of_edges()} edges")
# print(f"Illicit addresses after structural filter: {len(illicit_in_filtered_1)} out of {len(illicit_set)}")

# === Step 2: Remove edges with unknown addresses ===
""" Keep only edges where both endpoints are known address IDs """
df_filtered = df[
    df['addr_id1'].isin(known_ids) &
    df['addr_id2'].isin(known_ids)
]

# Rebuild the graph with the filtered DataFrame
G = build_graph(df_filtered)  

# --- Check illicit addresses in the filtered graph ---
illicit_in_filtered_2 = illicit_set & set(G.nodes) 
print(f"\nAfter removing unknown addresses: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Illicit addresses after full filtering: {len(illicit_in_filtered_2)} out of {len(illicit_set)}")

# --- Extract k-hop clusters ---
""" Extract k-hop clusters from the filtered DataFrame """
k = 1
subgraphs, sizes = extract_k_hop_illicit_clusters(df_filtered, illicit_set, k=1) 
print(f"\n Extracted {len(subgraphs)} non-overlapping {k}-hop clusters.")
print("Top-5 cluster sizes:", sorted(sizes, reverse=True)[:5])
plot_khop_cluster_size_distribution(subgraphs)



# --- Compute cluster stats with triangles ---
print("\nComputing cluster stats with triangles...")
cluster_stats = []
for cluster_id, g in enumerate(subgraphs):
    nodes = set(g.nodes())
    illicit_nodes = nodes & illicit_set
    num_illicit = len(illicit_nodes)
    if num_illicit == 0:
        continue
    num_normal = len(nodes) - num_illicit
    total = len(nodes)
    ratio = num_illicit / total
    triangles = count_triangles(g)
    tri_density = compute_tri_density(g)  # Correct TriDensity calculation
    cluster_stats.append((cluster_id, num_illicit, num_normal, total, ratio, triangles, tri_density))

# Sort by number of illicit nodes
cluster_stats = sorted(cluster_stats, key=lambda x: x[1], reverse=True)[:250]

print(f"{'ClusterID':<10}{'#Illicit':<10}{'#Normal':<10}{'Total':<10}{'Illicit%':<10}{'#Triangles':<12}{'TriDensity':<12}")
for (cluster_id, num_illicit, num_normal, total, ratio, triangles, tri_density) in cluster_stats:
    print(f"{cluster_id:<10}{num_illicit:<10}{num_normal:<10}{total:<10}{ratio * 100:<10.1f}{triangles:<12}{tri_density:<12.3f}")

# --- Check node overlap between clusters 43 and 66 (example) ---
G_43_nodes = set(subgraphs[43].nodes())
G_66_nodes = set(subgraphs[66].nodes())
common_nodes = G_43_nodes & G_66_nodes

print(f"\nCluster 43 node count: {len(G_43_nodes)}")
print(f"Cluster 66 node count: {len(G_66_nodes)}")
print(f"Number of common nodes: {len(common_nodes)}")
print("Example common node IDs:", list(common_nodes)[:10])

# --- Count total unique nodes in illicit clusters ---
illicit_cluster_nodes = set()
for g in subgraphs:
    nodes = set(g.nodes)
    if nodes & illicit_set:  
        illicit_cluster_nodes.update(nodes)
print(f"\nTotal unique nodes in illicit clusters: {len(illicit_cluster_nodes)}")

# --- Cluster-level illicit statistics ---
total_clusters = len(subgraphs)
illicit_cluster_ids = [i for i, g in enumerate(subgraphs) if len(set(g.nodes) & illicit_set) > 0]
illicit_clusters_count = len(illicit_cluster_ids)

print(f"\nTotal number of extracted clusters: {len(subgraphs)}")
print(f"Number of clusters with at least one illicit node: {illicit_clusters_count}")
print(f"Percentage of illicit clusters: {illicit_clusters_count / len(subgraphs) * 100:.1f}%")
