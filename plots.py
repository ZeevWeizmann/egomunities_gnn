"""
Plot average F1-score (illicit) across tri_density bins
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# --- Load results ---
df = pd.read_excel("manual_pair_results_all_models.xlsx")

# --- Add ratio of illicit nodes ---
df["illicit_ratio"] = df["support_illicit"] / df["n_nodes"]

# --- Filter: focus on mid-range tri_density 
df = df[(df["tri_density"] >= 0.04) & (df["tri_density"] <= 0.058)]

# --- Bin tri_density into 10 quantile bins (within [0.04, 0.058]) ---
df["tri_bin"] = pd.qcut(df["tri_density"], q=10, duplicates="drop")

# --- Compute mean F1-score per bin for each model ---
df_grouped = (
    df.groupby(["tri_bin", "model"])["f1_illicit"]
    .mean()
    .reset_index()
)

# --- Add numeric index for plotting ---
df_grouped["bin_idx"] = df_grouped["tri_bin"].cat.codes + 1

# --- Save actual bin labels ---
bin_labels = df_grouped["tri_bin"].cat.categories.astype(str)

# --- Plot ---
plt.figure(figsize=(14,6))
sns.lineplot(
    data=df_grouped,
    x="bin_idx",
    y="f1_illicit",
    hue="model",
    marker="o"
)

plt.title("Average F1-score (illicit) for tri_density range [0.04–0.058] (10 bins)")
plt.xlabel("tri_density bins")
plt.ylabel("Average F1-score (illicit)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Model")

# --- Replace X-axis with actual bin ranges ---
plt.xticks(
    ticks=range(1, len(bin_labels)+1),
    labels=bin_labels,
    rotation=45,
    ha="right"
)

plt.tight_layout()
plt.show()
