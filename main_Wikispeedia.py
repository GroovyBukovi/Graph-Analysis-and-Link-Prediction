import collections
from networkx.algorithms.community import greedy_modularity_communities
import pandas as pd
from urllib.parse import unquote
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from community import community_louvain
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from community import community_louvain
import networkx as nx

# STEP 1: Load data
data_folder = "wikispeedia_paths-and-graph"

articles_df = pd.read_csv(f"{data_folder}/articles.tsv", sep="\t", header=None, names=["Article"])
links_df = pd.read_csv(f"{data_folder}/links.tsv", sep="\t", comment="#", header=None, names=["From", "To"])
paths_df = pd.read_csv(f"{data_folder}/paths_finished.tsv", sep="\t", comment="#", header=None,
                       names=["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"])

# STEP 2: Decode article names (URL decoding)
links_df["From"] = links_df["From"].apply(unquote)
links_df["To"] = links_df["To"].apply(unquote)

# STEP 3: Build directed graph
G = nx.from_pandas_edgelist(links_df, source="From", target="To", create_using=nx.DiGraph())

print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# STEP 4: Structural analysis
print("\n--- Structural Analysis ---")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Strongly Connected Components:", nx.number_strongly_connected_components(G))
print("Weakly Connected Components:", nx.number_weakly_connected_components(G))

# Density: how tightly connected the graph is
print("Graph Density:", nx.density(G))

# Assortativity: correlation between degrees of connected nodes
print("Degree Assortativity:", nx.degree_assortativity_coefficient(G))

# Reciprocity: how often links are bidirectional
print("Reciprocity:", nx.reciprocity(G))

"""# STEP 5: Centrality analysis
print("\n--- Centrality Measures ---")
# In-degree and out-degree
in_deg = nx.in_degree_centrality(G)
out_deg = nx.out_degree_centrality(G)"""

# PageRank (better than eigenvector for directed)
pagerank = nx.pagerank(G)

"""# Betweenness
bet_cent = nx.betweenness_centrality(G)

# Closeness (use directed)
clo_cent = nx.closeness_centrality(G)
"""
# Top 5 nodes by PageRank
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 by PageRank:")
for node, score in top_pagerank:
    print(f"{node}: {score:.4f}")

print("\n--- In-Degree (Top 10 Nodes) ---")
in_degrees = dict(G.in_degree())
top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
for node, degree in top_in:
    print(f"{node}: in-degree = {degree}")

print("\n--- Out-Degree (Top 10 Nodes) ---")
out_degrees = dict(G.out_degree())
top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
for node, degree in top_out:
    print(f"{node}: out-degree = {degree}")

# STEP 6: Visualize a subgraph

sample_nodes = list(G.nodes)[:100]
subG = G.subgraph(sample_nodes)
"""
plt.figure(figsize=(40, 32))
nx.draw(subG.to_undirected(), node_size=300, node_color='black', edge_color='gray', with_labels=False)
plt.title("Wikispeedia Subgraph (100 nodes)")
plt.savefig("wikispeedia_subgraph.png", dpi=300)
plt.close()

############ COMMUNITIES ############
print("Saved subgraph visualization.")



# Load categories.tsv
category_df = pd.read_csv("wikispeedia_paths-and-graph/categories.tsv", sep="\t", header=None, names=["Article", "Category"])
category_map = dict(zip(category_df["Article"], category_df["Category"]))

# Community detection (use undirected version)
print("\n--- Top 5 Community Themes ---")
undirected_G = G.to_undirected()
communities = list(greedy_modularity_communities(undirected_G))

for i, community in enumerate(communities[:5]):
    community_articles = list(community)
    themes = []

    for article in community_articles:
        if article in category_map:
            themes.append(category_map[article])

    if themes:
        top_theme = collections.Counter(themes).most_common(1)[0]
        print(f"Community {i+1} (size: {len(community)}): Top theme → {top_theme[0]} (count: {top_theme[1]})")
    else:
        print(f"Community {i+1} (size: {len(community)}): No category data found.")


# STEP 3B: Build user behavior-based directed graph
print("\n--- Building Behavior Graph from User Paths ---")
G_behavior = nx.DiGraph()

# Parse each path and add edges based on user transitions
for path in paths_df["path"].dropna():
    articles = path.split(";")
    for i in range(len(articles) - 1):
        src = unquote(articles[i])
        dst = unquote(articles[i + 1])
        G_behavior.add_edge(src, dst)

print(f"Behavior Graph built with {G_behavior.number_of_nodes()} nodes and {G_behavior.number_of_edges()} edges.")

#######

print("\n--- Behavioral Analysis from Paths ---")

# Drop rows with missing paths
paths_df = paths_df.dropna(subset=["path"])

# Parse paths
paths_df["ArticlePath"] = paths_df["path"].apply(lambda x: x.split(";"))

# Compute basic stats
total_paths = len(paths_df)
avg_length = paths_df["ArticlePath"].apply(len).mean()
avg_duration = paths_df["durationInSec"].mean()

print(f"Total navigation paths: {total_paths}")
print(f"Average path length (in hops): {avg_length:.2f}")
print(f"Average time spent (in seconds): {avg_duration:.2f}")

# Most common start and end articles
start_counts = paths_df["ArticlePath"].apply(lambda x: x[0]).value_counts()
end_counts = paths_df["ArticlePath"].apply(lambda x: x[-1]).value_counts()

print("\nTop 5 Starting Articles:")
for article, count in start_counts.head(5).items():
    print(f"{article}: {count} paths")

print("\nTop 5 Ending Articles:")
for article, count in end_counts.head(5).items():
    print(f"{article}: {count} paths")

# Top visited articles overall (not just start/end)
from collections import Counter
all_articles = paths_df["ArticlePath"].explode()
top_visited = Counter(all_articles).most_common(5)

print("\nTop 5 Most Visited Articles Overall:")
for article, count in top_visited:
    print(f"{article}: {count} appearances")

# If ratings are available
if "rating" in paths_df.columns and pd.api.types.is_numeric_dtype(paths_df["rating"]):
    valid_ratings = paths_df["rating"].dropna()
    if not valid_ratings.empty:
        print(f"\nAverage user rating: {valid_ratings.mean():.2f}")
        print("Rating distribution:")
        print(valid_ratings.value_counts().sort_index())


#### How efficintly users navigate the path

print("\n--- Path Efficiency Analysis ---")

shortest_lengths = []
actual_lengths = []

for path in paths_df["ArticlePath"]:
    if len(path) < 2:
        continue
    start, end = path[0], path[-1]
    try:
        shortest = nx.shortest_path_length(G, source=start, target=end)
        actual = len(path) - 1
        shortest_lengths.append(shortest)
        actual_lengths.append(actual)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        continue

if shortest_lengths:
    efficiency = [s / a for s, a in zip(shortest_lengths, actual_lengths) if a != 0]
    print(f"Average efficiency (shortest/actual): {sum(efficiency)/len(efficiency):.3f}")
    print(f"Average detour (actual - shortest): {sum(a - s for s, a in zip(shortest_lengths, actual_lengths)) / len(shortest_lengths):.2f}")

bigrams = Counter()
for path in paths_df["ArticlePath"]:
    bigrams.update(zip(path[:-1], path[1:]))

print("\nTop 5 Most Common Article Transitions:")
for (a, b), count in bigrams.most_common(5):
    print(f"{a} → {b}: {count} times")

from collections import defaultdict

end_path_map = defaultdict(list)
for row in paths_df.itertuples():
    if not isinstance(row.ArticlePath, list):
        continue
    end = row.ArticlePath[-1]
    end_path_map[end].append(tuple(row.ArticlePath))

most_common_ends = sorted(end_path_map.items(), key=lambda x: len(x[1]), reverse=True)[:3]

print("\nTop 3 End Articles by Number of Unique Paths:")
for end, paths in most_common_ends:
    unique_paths = len(set(paths))
    print(f"{end}: {unique_paths} unique paths (out of {len(paths)} total)")



# STEP X: Path Length vs. Average Duration
print("\n--- Path Length vs. Average Duration ---")

# Clean and split paths
paths_df = paths_df.copy()
paths_df["PathList"] = paths_df["path"].dropna().apply(lambda x: x.split(";"))

# Calculate path lengths and durations
paths_df["PathLength"] = paths_df["PathList"].apply(len)
paths_df["durationInSec"] = pd.to_numeric(paths_df["durationInSec"], errors='coerce')

# Group by path length and calculate average duration
avg_durations = paths_df.groupby("PathLength")["durationInSec"].mean()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(avg_durations.index, avg_durations.values, marker='o')
plt.title("Average Duration vs. Path Length")
plt.xlabel("Path Length (Number of Articles)")
plt.ylabel("Average Duration (seconds)")
plt.grid(True)
plt.tight_layout()
plt.savefig("average_duration_vs_path_length.png", dpi=300)
plt.show()
"""

######################### LOUVAIN ##########################
"""


# Load categories
category_df = pd.read_csv(f"{data_folder}/categories.tsv", sep="\t", header=None, names=["Article", "Category"])
category_map = dict(zip(category_df["Article"], category_df["Category"]))

# Convert to undirected graph for community detection
undirected_G = G.to_undirected()

print("\n--- Louvain Community Analysis ---")
# 1. Compute Louvain partition (node → community id)
partition = community_louvain.best_partition(undirected_G)

# 2. Invert partition: community_id → set of nodes
community_nodes = {}
for node, comm_id in partition.items():
    community_nodes.setdefault(comm_id, set()).add(node)

# 3. Sort communities by size
sorted_communities = sorted(community_nodes.items(), key=lambda x: len(x[1]), reverse=True)

# 4. Analyze top 5 communities
for i, (comm_id, nodes) in enumerate(sorted_communities[:5], start=1):
    subgraph = undirected_G.subgraph(nodes)
    size = len(nodes)
    density = nx.density(subgraph)
    clustering = nx.average_clustering(subgraph)

    # Top theme
    themes = [category_map[node] for node in nodes if node in category_map]
    if themes:
        top_theme, count = Counter(themes).most_common(1)[0]
    else:
        top_theme, count = "N/A", 0

    # Top nodes (by degree)
    degrees = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)[:3]
    top_nodes = [node for node, _ in degrees]

    print(f"\nCommunity {i}:")
    print(f"- Size: {size}")
    print(f"- Top theme: {top_theme} (count: {count})")
    print(f"- Density: {density:.3f}")
    print(f"- Average clustering: {clustering:.2f}")
    print(f"- Top nodes: {', '.join(top_nodes)}")


############### SPECTRAL CLUSTERING ###################


# --- Load data ---
data_folder = "wikispeedia_paths-and-graph"
articles_df = pd.read_csv(f"{data_folder}/articles.tsv", sep="\t", header=None, names=["Article"])
links_df = pd.read_csv(f"{data_folder}/links.tsv", sep="\t", comment="#", header=None, names=["From", "To"])
category_df = pd.read_csv(f"{data_folder}/categories.tsv", sep="\t", header=None, names=["Article", "Category"])

links_df["From"] = links_df["From"].apply(unquote)
links_df["To"] = links_df["To"].apply(unquote)
category_map = dict(zip(category_df["Article"], category_df["Category"]))

# --- Build undirected graph ---
G = nx.from_pandas_edgelist(links_df, source="From", target="To")
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# --- Filter largest connected component ---
largest_cc = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(largest_cc).copy()

# --- Spectral clustering ---
print("\n--- Performing Spectral Clustering ---")
adj_matrix = nx.to_numpy_array(Gcc)
n_clusters = 5

spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
labels = spectral.fit_predict(adj_matrix)

# --- Assign nodes to clusters ---
node_list = list(Gcc.nodes())
clusters = [[] for _ in range(n_clusters)]
for node, label in zip(node_list, labels):
    clusters[label].append(node)

# --- Describe each cluster ---
print("\n--- Top 5 Communities via Spectral Clustering ---")
for i, community_nodes in enumerate(clusters):
    community_subgraph = Gcc.subgraph(community_nodes)

    # Theme analysis
    themes = [category_map[node] for node in community_nodes if node in category_map]
    top_theme = Counter(themes).most_common(1)
    theme_str = f"{top_theme[0][0]} (count: {top_theme[0][1]})" if top_theme else "No category data"

    # Density and clustering
    density = nx.density(community_subgraph)
    clustering = nx.average_clustering(community_subgraph)

    # Top nodes by degree
    degrees = dict(community_subgraph.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:3]

    print(f"\nCommunity {i + 1}:")
    print(f"- Size: {len(community_nodes)}")
    print(f"- Top theme: {theme_str}")
    print(f"- Density: {density:.3f}")
    print(f"- Average clustering: {clustering:.2f}")
    print(f"- Top nodes: {', '.join(top_nodes)}")


# === 3. Wikispeedia: Node Size = PageRank, Color = Degree ===



def visualize_wikispeedia_clean(G, pagerank=None, top_n=300, path="wikispeedia_clean_graph.png"):
    if pagerank is None:
        pagerank = nx.pagerank(G)

    # Select top nodes by PageRank
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
    sub_nodes = [n for n, _ in top_nodes]
    subG = G.subgraph(sub_nodes).copy()

    pagerank_sub = {n: pagerank[n] for n in subG.nodes()}
    degree_sub = dict(subG.degree())
    pos = nx.kamada_kawai_layout(subG)

    # Normalize PageRank to node size
    pagerank_vals = np.array(list(pagerank_sub.values()))
    sizes = MinMaxScaler((100, 1000)).fit_transform(pagerank_vals.reshape(-1, 1)).flatten()

    # Normalize degree to node color
    degree_vals = np.array(list(degree_sub.values()))
    colors = MinMaxScaler((0, 1)).fit_transform(degree_vals.reshape(-1, 1)).flatten()
    cmap = cm.viridis

    # Start plot
    fig, ax = plt.subplots(figsize=(20, 15))

    # Draw nodes and edges
    nodes = nx.draw_networkx_nodes(
        subG, pos, node_size=sizes, node_color=colors, cmap=cmap, alpha=0.85, ax=ax
    )
    nx.draw_networkx_edges(subG, pos, alpha=0.3, width=0.7, edge_color='gray', ax=ax)

    # Add colorbar for degree
    norm = mcolors.Normalize(vmin=degree_vals.min(), vmax=degree_vals.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Node Degree", fontsize=12)

    # Final touches
    ax.set_title("Top Wikispeedia Nodes\n(Size = PageRank, Color = Degree)", fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

pagerank = nx.pagerank(G)
visualize_wikispeedia_clean(G, pagerank, top_n=500)"""








"""import networkx as nx

def find_flow_bottleneck_edges_with_unit_capacity(G, source, target):
    # Create a copy with unit capacities
    G_cap = G.copy()
    for u, v in G_cap.edges():
        G_cap[u][v]["capacity"] = 1

    try:
        cut_value, partition = nx.minimum_cut(G_cap, source, target)
        reachable, non_reachable = partition

        cutset = set()
        for u in reachable:
            for v in G_cap[u]:
                if v in non_reachable:
                    cutset.add((u, v))

        print(f"\n--- Flow Bottleneck Analysis (Unit Capacity) ---")
        print(f"Source: {source}")
        print(f"Target: {target}")
        print(f"Min-cut value: {cut_value}")
        print("Edges whose removal would block all flow:")
        for u, v in cutset:
            print(f"{u} → {v}")

    except nx.NetworkXError as e:
        print(f"NetworkX Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")




(find_flow_bottleneck_edges_with_unit_capacity(G, source="United_States", target="Europe"))"""

import networkx as nx

# Use the undirected version of the graph for articulation point analysis
G_undirected = G.to_undirected()

# Find articulation points
articulation_points = list(nx.articulation_points(G_undirected))
print(f"\n--- Articulation Points ---")
print(f"Total articulation points: {len(articulation_points)}")

# Top articulation points by degree (to filter down to meaningful ones)
degrees = dict(G_undirected.degree(articulation_points))
top_articulations = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 articulation points (by degree):")
for node, degree in top_articulations:
    print(f"{node}: degree = {degree}")

import networkx as nx
import matplotlib.pyplot as plt


import networkx as nx
import matplotlib.pyplot as plt

# Use the largest connected component for better visuals
G_ud = G.to_undirected()
largest_cc = max(nx.connected_components(G_ud), key=len)
Gcc = G_ud.subgraph(largest_cc).copy()

# Find articulation points
articulation_points = set(nx.articulation_points(Gcc))

# Create color and size maps
node_colors = []
node_sizes = []

for node in Gcc.nodes():
    if node in articulation_points:
        node_colors.append("red")
        node_sizes.append(80)
    else:
        node_colors.append("lightgray")
        node_sizes.append(15)

# Draw graph
pos = nx.spring_layout(Gcc, seed=42)
plt.figure(figsize=(16, 14))
nx.draw(
    Gcc, pos,
    node_color=node_colors,
    node_size=node_sizes,
    edge_color="gray",
    width=0.5,
    alpha=0.6,
    with_labels=False
)
plt.title("Articulation Points in Wikispeedia Subgraph", fontsize=16)
plt.tight_layout()
plt.savefig("articulation_colored.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def visualize_betweenness_on_articulation_subgraph(G, top_n=300, top_k=50, path="betweenness_contrast_fixed.png"):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # Convert to undirected and extract top-N by degree
    G_undirected = G.to_undirected()
    top_nodes = sorted(G_undirected.degree, key=lambda x: x[1], reverse=True)[:top_n]
    sub_nodes = [n for n, _ in top_nodes]
    subG = G_undirected.subgraph(sub_nodes).copy()

    # Compute betweenness centrality on subgraph
    bet_cent = nx.betweenness_centrality(subG)

    # Identify top-K betweenness nodes
    top_bet_nodes = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_bet_set = set(n for n, _ in top_bet_nodes)

    # Normalize values
    top_values = np.array([bet_cent[n] for n in top_bet_set])
    scaler = MinMaxScaler((0.4, 1.0))
    normed_top = scaler.fit_transform(top_values.reshape(-1, 1)).flatten()
    cmap = plt.cm.YlOrRd
    color_map = dict(zip(top_bet_set, normed_top))

    # Set node colors and sizes
    node_colors = []
    node_sizes = []
    for n in subG.nodes():
        if n in top_bet_set:
            node_colors.append(cmap(color_map[n]))
            node_sizes.append(120)
        else:
            node_colors.append("lightgray")
            node_sizes.append(30)

    # Draw
    pos = nx.spring_layout(subG, seed=42)
    plt.figure(figsize=(18, 14))
    nx.draw_networkx_edges(subG, pos, alpha=0.2, width=0.4, edge_color="gray")
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes)

    # Optional: label top 5 betweenness nodes
    labeled = sorted(top_bet_set, key=lambda x: subG.degree(x), reverse=True)[:5]
    nx.draw_networkx_labels(subG, pos, labels={n: n for n in labeled}, font_size=10, font_color="blue")

    plt.title("Betweenness Centrality in Same Subgraph as Articulation Points", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved visualization to: {path}")


visualize_betweenness_on_articulation_subgraph(G)