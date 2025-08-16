
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("darknet_combined_labeled.csv")
print("✅ Shape:", df.shape)
print("✅ Columns:", df.columns.tolist())

print("\n✅ Missing values per column:")
print(df.isnull().sum())

print("\n✅ Label counts in the dataset:")
print(df['label'].value_counts())

print("\n📊 Top 20 UDP Destination Ports")
sns.countplot(data=df[df['udp.dstport'].notnull()], x='udp.dstport', order=df['udp.dstport'].value_counts().iloc[:20].index)
plt.title("Top 20 UDP Destination Ports")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n📊 Top 20 TCP Destination Ports")
sns.countplot(data=df[df['tcp.dstport'].notnull()], x='tcp.dstport', order=df['tcp.dstport'].value_counts().iloc[:20].index)
plt.title("Top 20 TCP Destination Ports")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n📊 Packet Size Distribution")
plt.figure(figsize=(10,6))
df['frame.len'].hist(bins=100)
plt.title("Packet Size Distribution")
plt.xlabel("Frame Length (Bytes)")
plt.ylabel("Frequency")
plt.show()

print("\n📊 Protocol Distribution")
sns.countplot(x='ip.proto', data=df)
plt.title("Protocol Distribution")
plt.show()
