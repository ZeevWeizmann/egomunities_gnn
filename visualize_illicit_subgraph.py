""" Visualize a specific egonet based on its index in the extracted subgraphs list from egonets_stats.py """

import matplotlib.pyplot as plt
import networkx as nx
from data_loader import load_graph_and_labels
from utils import filter_graph
from egonets import extract_k_hop_illicit_clusters

# --- Load and prepare ---
df, merged, illicit_set, addresses = load_graph_and_labels()
df_filtered = filter_graph(df)
df_filtered = df
# Keep only edges between known address_ids
""" Filter the DataFrame to keep only edges where both addr_id1 and addr_id2 are in known_ids """
known_ids = set(addresses['address_id'])
df_filtered = df_filtered[
    df_filtered['addr_id1'].isin(known_ids) &
    df_filtered['addr_id2'].isin(known_ids)
]
# --- Build main graph and extract subgraphs ---
""" Build the main graph from the filtered DataFrame """
subgraphs, sizes = extract_k_hop_illicit_clusters(df_filtered, illicit_set, k=1)

# --- Choose subgraph to visualize ---
""" Select a specific subgraph by index """
SUBGRAPH_INDEX = 776
  # Change this index to visualize different subgraphs
G = subgraphs[SUBGRAPH_INDEX].copy()

# --- Address mappings ---
# Create a mapping from address_id to address
id_to_address_all = dict(zip(addresses['address_id'], addresses['address']))

# Create a mapping from address_id to (address, illicit_category)
id_to_illicit_info = dict(zip(
    merged['address_id'],
    zip(merged['address'], merged['illicit_category'])
))

# --- Node colors and labels ---
""" Prepare node colors and labels for visualization """
node_colors = ['red' if node in illicit_set else 'skyblue' for node in G.nodes()]
node_sizes = [100 if node in illicit_set else 20 for node in G.nodes()]
graph_labels = {}
for node in G.nodes():
    if node in id_to_illicit_info:
        addr, category = id_to_illicit_info[node]
        label = f"{addr}\n({category})"
    elif node in id_to_address_all:
        label = id_to_address_all[node]
    else:
        label = str(node)
    graph_labels[node] = label

# --- Draw graph ---
""" Visualize the subgraph with addresses as labels """
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=SUBGRAPH_INDEX)

nx.draw(
    G,
    pos,
    with_labels=True,
    labels=graph_labels,
    node_color=node_colors,
    node_size=node_sizes,
    edge_color="gray",
    font_size=8
)

plt.title(f"Visualization of subgraph #{SUBGRAPH_INDEX} ")
plt.axis("off")
plt.tight_layout()
plt.show()

print(f"Number of edges in subgraph #{SUBGRAPH_INDEX}: {G.number_of_edges()}")

# Create a DataFrame for the component nodes
import pandas as pd
node_data = []
for node in G.nodes():
    address = id_to_address_all.get(node, None)
    illicit_category = id_to_illicit_info.get(node, ('Unknown', 'non-illicit'))[1]
    node_data.append((node, address, illicit_category))

node_df = pd.DataFrame(node_data, columns=['address_id', 'address', 'illicit_category'])
print("\nNode information for subgraph:")
print(node_df.to_string(index=False))

# --- Extra: print node lists and save CSV ---

# Save the node table to CSV
node_df_sorted = node_df.sort_values("address_id")
out_csv = f"subgraph_{SUBGRAPH_INDEX}_nodes.csv"
node_df_sorted.to_csv(out_csv, index=False)
print(f"\nNode table saved to: {out_csv}")
