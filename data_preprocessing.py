import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('darknet_combined_labeled.csv')

# Initial inspection
print("Initial shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Clean column names (optional)
df.columns = df.columns.str.strip()

# Fill mutually exclusive port fields with 0
df['tcp.srcport'] = df['tcp.srcport'].fillna(0)
df['tcp.dstport'] = df['tcp.dstport'].fillna(0)
df['udp.srcport'] = df['udp.srcport'].fillna(0)
df['udp.dstport'] = df['udp.dstport'].fillna(0)

# Fill protocol-level NaNs with mode or 0
df['ip.src'] = df['ip.src'].fillna('0.0.0.0')
df['ip.dst'] = df['ip.dst'].fillna('0.0.0.0')
df['ip.proto'] = df['ip.proto'].fillna(df['ip.proto'].mode()[0])

# Ensure label and frame.len are clean
assert df['label'].isnull().sum() == 0
assert df['frame.len'].isnull().sum() == 0

# Convert protocols to numeric (just in case)
df['ip.proto'] = df['ip.proto'].astype(int)
df['frame.len'] = df['frame.len'].astype(int)

# Save cleaned data
df.to_csv('darknet_cleaned.csv', index=False)
print("Cleaned data saved to darknet_cleaned.csv")
