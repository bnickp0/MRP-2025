import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"darknet_combined_labeled.csv"
df = pd.read_csv(file_path)




sns.set(style="whitegrid")


plt.figure(figsize=(8, 5))
sns.countplot(x="ip.proto", data=df)
plt.title("Protocol Distribution")
plt.xlabel("Protocol (ip.proto)")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df["frame.len"], bins=50, kde=True)
plt.title("Packet Size Distribution")
plt.xlabel("Frame Length (Bytes)")
plt.ylabel("Frequency")
plt.show()


top_tcp_ports = df["tcp.dstport"].dropna().value_counts().head(20)
top_udp_ports = df["udp.dstport"].dropna().value_counts().head(20)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_tcp_ports.index.astype(int), y=top_tcp_ports.values)
plt.title("Top 20 TCP Destination Ports")
plt.xlabel("TCP Dst Port")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=top_udp_ports.index.astype(int), y=top_udp_ports.values)
plt.title("Top 20 UDP Destination Ports")
plt.xlabel("UDP Dst Port")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

