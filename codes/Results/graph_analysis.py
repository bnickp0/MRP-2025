import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = 'MRP\Dataset\darknet_combined_labeled.csv'
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH, low_memory=False)


if 'ip.src' not in df.columns or 'ip.dst' not in df.columns:
    raise ValueError("Source or Destination IP columns not found in dataset!")

# Build graph from IP connections
print(" Building directed graph from source/destination IPs...")
edges = list(zip(df['ip.src'], df['ip.dst']))
G = nx.DiGraph()
G.add_edges_from(edges)

print(f" Nodes: {G.number_of_nodes()}")
print(f" Edges: {G.number_of_edges()}")


print("Extracting largest connected component...")
largest_cc = max(nx.weakly_connected_components(G), key=len)
G_largest = G.subgraph(largest_cc).copy()
print(f"Largest Component → {G_largest.number_of_nodes()} nodes, {G_largest.number_of_edges()} edges")

print("Plotting degree distribution...")
degrees = [G_largest.degree(n) for n in G_largest.nodes()]
plt.figure(figsize=(10, 6))
sns.histplot(degrees, bins=50, kde=True)
plt.title('Degree Distribution (Largest Component)')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("degree_distribution.png")
plt.close()

print(" Plotting connected component sizes...")
component_sizes = [len(c) for c in nx.weakly_connected_components(G)]
plt.figure(figsize=(10, 6))
sns.histplot(component_sizes, bins=30, log_scale=True)
plt.title('Connected Component Sizes')
plt.xlabel('Component Size')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("component_sizes.png")
plt.close()


print(" Drawing a representative subgraph...")
sub_nodes = list(G_largest.nodes())[:100]
subG = G_largest.subgraph(sub_nodes)

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(subG, seed=42)
nx.draw_networkx_nodes(subG, pos, node_size=20, alpha=0.7)
nx.draw_networkx_edges(subG, pos, alpha=0.3)
plt.title("Network Subgraph (100 nodes from largest component)")
plt.axis('off')
plt.tight_layout()
plt.savefig("representative_subgraph.png")
plt.close()

print(" Graph analysis completed. Plots saved:")
print("- degree_distribution.png")
print("- component_sizes.png")
print("- representative_subgraph.png")
