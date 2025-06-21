import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('darknet_cleaned.csv')

# Class distribution
sns.countplot(x='label', data=df)
plt.title('Label Distribution')
plt.savefig('plots/label_distribution.png')

# Protocol distribution
df['ip.proto'].value_counts().plot(kind='bar', title='Protocol Distribution')
plt.savefig('plots/protocol_distribution.png')

# Packet size distribution
sns.histplot(df['frame.len'], bins=100)
plt.title('Packet Size Distribution')
plt.xlim(0, 3000)
plt.savefig('plots/packet_size_distribution.png')

# Top TCP and UDP destination ports
top_tcp = df[df['tcp.dstport'] > 0]['tcp.dstport'].value_counts().nlargest(15)
top_tcp.plot(kind='bar', title='Top 15 TCP Destination Ports')
plt.savefig('plots/tcp_ports.png')
plt.clf()

top_udp = df[df['udp.dstport'] > 0]['udp.dstport'].value_counts().nlargest(15)
top_udp.plot(kind='bar', title='Top 15 UDP Destination Ports')
plt.savefig('plots/udp_ports.png')
