import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx
from collections import Counter
from community import community_louvain
from sklearn.cluster import SpectralClustering

# Set folder path
data_folder = "facebook"

# Discover ego user IDs
file_list = os.listdir(data_folder)
ego_ids = set(fname.split('.')[0] for fname in file_list if fname.endswith('.edges'))

# Load all data into dictionaries
edges_dict = {}
feat_dict = {}
egofeat_dict = {}
circles_dict = {}

for ego_id in ego_ids:
    try:
        edges = pd.read_csv(f"{data_folder}/{ego_id}.edges", sep=" ", header=None, names=["From", "To"])
        edges_dict[ego_id] = edges

        feat = pd.read_csv(f"{data_folder}/{ego_id}.feat", sep=" ", header=None)
        feat.index = feat.index.astype(int)
        feat_dict[ego_id] = feat

        egofeat = pd.read_csv(f"{data_folder}/{ego_id}.egofeat", sep=" ", header=None)
        egofeat_dict[ego_id] = egofeat

        circles_path = f"{data_folder}/{ego_id}.circles"
        if os.path.exists(circles_path):
            with open(circles_path, "r") as f:
                circles = [line.strip().split("\t") for line in f]
            circles_dict[ego_id] = circles
        else:
            circles_dict[ego_id] = []
    except Exception as e:
        print(f"Error loading ego {ego_id}: {e}")

# Load feature names once

ego_id = "107"  # Replace "0" with the ego ID you want to use
sample_id = ego_id
# Or comment out the line above to use the first ID automatically
if 'ego_id' not in locals():
    sample_id = next(iter(ego_ids))  # fallback automatic
    ego_id = sample_id

print(f"Using ego network with ID: {ego_id}")

with open(f"{data_folder}/{sample_id}.featnames", "r") as f:
    featnames = [line.strip() for line in f]


# Build one ego graph
ego_id = sample_id
friend_ids = feat_dict[ego_id].index.tolist()
G = nx.from_pandas_edgelist(edges_dict[ego_id], source="From", target="To")

# Add features to friends
for i, node_id in enumerate(friend_ids):
    G.add_node(node_id)
    G.nodes[node_id]['features'] = feat_dict[ego_id].iloc[i].tolist()

# Add ego node
ego_node = int(ego_id)
G.add_node(ego_node)
G.nodes[ego_node]['features'] = egofeat_dict[ego_id].iloc[0].tolist()

# Connect ego to all friends
for friend in friend_ids:
    G.add_edge(ego_node, friend)

print(f"Graph for ego {ego_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")


########################################################################

############ BRIDGE NODES ############


bridge_nodes = list(nx.articulation_points(G))
print("Bridge nodes:", bridge_nodes)

# Structural analysis
print("\n--- Structural Analysis ---")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected Components:", nx.number_connected_components(G))
print("Average Clustering Coefficient:", nx.average_clustering(G))

degrees = [d for _, d in G.degree()]
plt.hist(degrees, bins=20)
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()

# STEP 7: Centrality measures
print("\n--- Centrality Measures ---")
deg_cent = nx.degree_centrality(G)
bet_cent = nx.betweenness_centrality(G)
clo_cent = nx.closeness_centrality(G)
eig_cent = nx.eigenvector_centrality(G, max_iter=1000)

top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by Degree Centrality:")
for node, score in top_deg:
    print(f"Node {node}: {score:.4f}")

top_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by Betweenness Centrality:")
for node, score in top_bet:
    print(f"Node {node}: {score:.4f}")

top_clo = sorted(clo_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by Closeness Centrality:")
for node, score in top_clo:
    print(f"Node {node}: {score:.4f}")

top_eig = sorted(eig_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by Eigenvector Centrality:")
for node, score in top_eig:
    print(f"Node {node}: {score:.4f}")


# Save visualization to file
plt.figure(figsize=(20, 16))
node_colors = ['green' if node == ego_node else 'black' for node in G.nodes()]
node_sizes = [250 if node == ego_node else 30 for node in G.nodes()]
nx.draw(G, node_size=node_sizes, node_color=node_colors, # Already defined earlier
    edge_color='gray',      # Softer edges
    width=0.3,              # Thinner lines
    with_labels=False)
plt.title(f"Ego Network for User {ego_id}")
plt.savefig(f"ego_graph_{ego_id}.png", dpi=300)
plt.close()

print(f"Graph image saved as ego_graph_{ego_id}.png")

################### Feature Similarity Graph ####################

# === SIMILARITY-BASED GRAPH (≥70% overlap with ego's features) ===
print("\n--- Building Shared Feature Similarity Graph (≥15% match) ---")

# Get ego features (binary vector)
ego_feat = egofeat_dict[ego_id].iloc[0].values
ego_feat_indices = np.where(ego_feat == 1)[0]  # Indices where ego has feature "on"

# If ego has no active features, skip
if len(ego_feat_indices) == 0:
    print("Ego has no active features. Skipping similarity-based graph.")
else:
    G_sim = nx.Graph()
    G_sim.add_node(ego_id, color='green')

    for friend_id in feat_dict[ego_id].index:
        friend_feat = feat_dict[ego_id].loc[friend_id].values
        overlap = friend_feat[ego_feat_indices].sum()
        match_ratio = overlap / len(ego_feat_indices)

        G_sim.add_node(friend_id, color='blue')
        if match_ratio > 0.15:
            G_sim.add_edge(ego_id, friend_id, weight=match_ratio)

    print(f"Constructed similarity graph with {G_sim.number_of_nodes()} nodes and {G_sim.number_of_edges()} edges.")

    print("\n--- Building Shared Feature Similarity Graph (≥70% match) ---")

    # Get ego features (binary vector)
    ego_feat = egofeat_dict[ego_id].iloc[0].values
    ego_feat_indices = np.where(ego_feat == 1)[0]  # Indices where ego has feature "on"

    # If ego has no active features, skip
    if len(ego_feat_indices) == 0:
        print("Ego has no active features. Skipping similarity-based graph.")
    else:
        G_sim = nx.Graph()
        G_sim.add_node(ego_id)

        for friend_id in feat_dict[ego_id].index:
            friend_feat = feat_dict[ego_id].loc[friend_id].values
            overlap = friend_feat[ego_feat_indices].sum()
            match_ratio = overlap / len(ego_feat_indices)

            G_sim.add_node(friend_id)
            if match_ratio > 0.3:
                G_sim.add_edge(ego_id, friend_id, weight=match_ratio)

        print(f"Constructed similarity graph with {G_sim.number_of_nodes()} nodes and {G_sim.number_of_edges()} edges.")

        # Assign node colors
        colors = []
        for node in G_sim.nodes():
            if node == ego_id:
                colors.append("forestgreen")
            elif G_sim.has_edge(ego_id, node):
                colors.append("lightgreen")
            else:
                colors.append("grey")

        sizes = [180 if node == ego_id else 30 for node in G_sim.nodes()]
        pos = nx.spring_layout(G_sim, seed=42, k=0.15)

        # Draw graph
        plt.figure(figsize=(10, 8))
        nx.draw(G_sim, pos, node_color=colors, node_size=sizes, with_labels=False, edge_color="black", width=0.5)
        plt.title(f"Ego Network (≥70% feature match with ego {ego_id})")
        plt.savefig(f"ego_shared_features_graph_{ego_id}.png", dpi=300)
        plt.close()

        print(f"Saved similarity graph as ego_shared_features_graph_{ego_id}.png")

############# Intuitions on Communities ##################


# GREEDY MODULARITY COMMUNITIES
print("\n--- Community Detection ---")
communities = list(greedy_modularity_communities(G))
print(f"Detected {len(communities)} communities.")
print("Community sizes:", [len(c) for c in communities])


print("\n--- Analyzing Most Common Anonymized Feature Values in Largest Community ---")

# Step 1: Largest community
largest_community = max(communities, key=len)
print(f"Largest community size: {len(largest_community)}")

expected_length = len(featnames)
valid_vectors = []

for node in largest_community:
    features = G.nodes[node].get('features')
    if features is not None:
        if len(features) < expected_length:
            # Fill with zeros if feature vector is too short
            padded = features + [0] * (expected_length - len(features))
            valid_vectors.append(padded)
        else:
            valid_vectors.append(features[:expected_length])


# Convert to NumPy array
feature_matrix = np.array(valid_vectors)

# Convert all non-zero values to 1 (binary presence)
binary_matrix = (feature_matrix != 0).astype(int)

# Sum presence across all users
feature_counts = binary_matrix.sum(axis=0)

# Step 4: Get top features
top_features = sorted(enumerate(feature_counts), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 most common anonymized features in the largest community:")
for idx, count in top_features:
    print(f"{idx}: {featnames[idx]} (count: {int(count)})")



# Feature exclusion settings
excluded_keywords = {"birthday"}

# Louvain community detection (on undirected graph)
print("\n--- Louvain Community Analysis ---")
partition = community_louvain.best_partition(G)  # G is your ego graph
communities_dict = {}
for node, comm_id in partition.items():
    communities_dict.setdefault(comm_id, set()).add(node)

# Sort by community size
top_communities = sorted(communities_dict.items(), key=lambda x: len(x[1]), reverse=True)[:10]

for i, (comm_id, comm_nodes) in enumerate(top_communities):
    comm_nodes = list(comm_nodes)

    # Feature vector aggregation
    vectors = []
    for node in comm_nodes:
        feats = G.nodes[node].get("features")
        if feats:
            vectors.append(feats)

    if not vectors:
        print(f"Community {i + 1}: No feature data.")
        continue

    expected_len = len(featnames)
    padded_vectors = []

    for v in vectors:
        if len(v) < expected_len:
            v = v + [0] * (expected_len - len(v))  # pad with zeros
        elif len(v) > expected_len:
            v = v[:expected_len]  # truncate extra values
        padded_vectors.append(v)

    feature_matrix = np.array(padded_vectors)

    binary_matrix = (feature_matrix != 0).astype(int)
    feature_counts = binary_matrix.sum(axis=0)

    # Find valid top feature
    sorted_features = sorted(enumerate(feature_counts), key=lambda x: x[1], reverse=True)
    top_feat_name = "N/A"
    top_feat_count = 0
    for idx, count in sorted_features:
        if idx < len(featnames):
            name = featnames[idx]
            if not any(excl in name.lower() for excl in excluded_keywords):
                top_feat_name = name
                top_feat_count = int(count)
                break

    # Density & clustering
    subG = G.subgraph(comm_nodes)
    density = nx.density(subG)
    clustering = nx.average_clustering(subG)

    # Top degree nodes
    degrees = subG.degree()
    top_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)[:3]
    top_node_names = [str(node) for node, _ in top_nodes]

    # Print results
    print(f"\nCommunity {i + 1}:")
    print(f"- Size: {len(comm_nodes)}")
    print(f"- Top feature: {top_feat_name} (count: {top_feat_count})")
    print(f"- Density: {density:.3f}")
    print(f"- Average clustering: {clustering:.2f}")
    print(f"- Top nodes: {', '.join(top_node_names)}")


# === 2. Ego Network with Louvain Communities ===
def draw_ego_louvain(G, ego_node_id, path="ego_network_community_colored.png"):
    partition = community_louvain.best_partition(G)
    pos = nx.spring_layout(G, seed=42)

    communities = set(partition.values())
    cmap = cm.get_cmap('tab20', len(communities))
    node_colors = [cmap(partition[n]) for n in G.nodes()]

    node_sizes = [300 if n == ego_node_id else 30 for n in G.nodes()]
    node_colors = ['green' if n == ego_node_id else cmap(partition[n]) for n in G.nodes()]

    plt.figure(figsize=(20, 16))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    plt.title(f"Ego Network (Louvain Communities)")
    plt.axis('off')
    plt.savefig(path, dpi=300)
    plt.close()

draw_ego_louvain(G, "107")

################# SPECTRAL CLUSTERING ####################


# Convert the NetworkX graph to an adjacency matrix
adj_matrix = nx.to_numpy_array(G)

# Define number of clusters (you can tune this)
k = 5

# Apply Spectral Clustering
print(f"\n--- Spectral Clustering with {k} clusters ---")
spectral = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=42)
labels = spectral.fit_predict(adj_matrix)

# Assign cluster labels to each node
node_list = list(G.nodes())
node_cluster_map = dict(zip(node_list, labels))

# Group nodes by cluster
clusters = [[] for _ in range(k)]
for node, cluster_id in node_cluster_map.items():
    clusters[cluster_id].append(node)

# Analyze each cluster
for i, cluster_nodes in enumerate(clusters):
    subG = G.subgraph(cluster_nodes)
    density = nx.density(subG)
    clustering = nx.average_clustering(subG)

    print(f"\nCluster {i + 1}:")
    print(f"- Size: {len(cluster_nodes)}")
    print(f"- Density: {density:.3f}")
    print(f"- Average Clustering: {clustering:.3f}")

    # Top degree nodes in this cluster
    degs = subG.degree()
    top_nodes = sorted(degs, key=lambda x: x[1], reverse=True)[:3]
    top_node_str = ', '.join([str(n) for n, _ in top_nodes])
    print(f"- Top nodes: {top_node_str}")

print("\n--- Feature Analysis for Each Cluster ---")

print("\n--- Top Features Across All Spectral Clusters ---")

# Assume featnames is already loaded and expected_length is known
expected_length = len(featnames)
all_vectors = []

for cluster_nodes in clusters:
    for node in cluster_nodes:
        features = G.nodes[node].get('features')
        if features is not None:
            # Ensure uniform length
            if len(features) < expected_length:
                features = features + [0] * (expected_length - len(features))
            elif len(features) > expected_length:
                features = features[:expected_length]
            all_vectors.append(features)

if all_vectors:
    feature_matrix = np.array(all_vectors)
    binary_matrix = (feature_matrix != 0).astype(int)
    feature_counts = binary_matrix.sum(axis=0)

    # Sort and print top N features
    top_features = sorted(enumerate(feature_counts), key=lambda x: x[1], reverse=True)

    print("Top 10 features among all clusters:")
    for idx, count in top_features[:10]:
        fname = featnames[idx]
        if "birthday" not in fname.lower():  # Optional: exclude birthdays
            print(f"- {fname} (count: {int(count)})")
else:
    print("No valid feature vectors found in clusters.")
