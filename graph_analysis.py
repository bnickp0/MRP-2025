import pandas as pd
import networkx as nx

df = pd.read_csv('darknet_cleaned.csv')

# Build directed graph from IPs
edges = df[['ip.src', 'ip.dst']].dropna().drop_duplicates()
G = nx.from_pandas_edgelist(edges, source='ip.src', target='ip.dst', create_using=nx.DiGraph())

# Basic network stats
print(nx.info(G))
print("Average degree:", sum(dict(G.degree()).values()) / G.number_of_nodes())
print("Connected components:", nx.number_weakly_connected_components(G))

# Save graph structure
nx.write_gexf(G, 'graph_traffic.gexf')
