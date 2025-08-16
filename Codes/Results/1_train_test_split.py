import pandas as pd
from sklearn.model_selection import train_test_split
import os


dataset_path = r"darknet_combined_labeled.csv"
output_path = r"MRP\Dataset\split"
os.makedirs(output_path, exist_ok=True)


df = pd.read_csv(dataset_path, low_memory=False)

# Fix the ip.proto conversion issue
df['ip.proto'] = df['ip.proto'].astype(str).str.replace(",", ".", regex=False)
df['ip.proto'] = pd.to_numeric(df['ip.proto'], errors='coerce').fillna(0).astype(int)

df['frame.len'] = df['frame.len'].fillna(0).astype(float)


label_map = {'Tor': 0, 'Freenet': 1, 'I2P': 2, 'Zeronet': 3}
df['label'] = df['label'].map(label_map)


df = df.dropna(subset=['ip.proto', 'frame.len', 'label'])


X = df[['ip.proto', 'frame.len']]
y = df['label']


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv(os.path.join(output_path, "train.csv"), index=False)
X_val.to_csv(os.path.join(output_path, "val.csv"), index=False)
y_train.to_csv(os.path.join(output_path, "train_labels.csv"), index=False)
y_val.to_csv(os.path.join(output_path, "val_labels.csv"), index=False)

print(" Done splitting and saving the datasets (stratified).")
