import pandas as pd
import joblib
import xgboost as xgb

print("🔄 Loading files...")
train = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\train.csv")
train_labels = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\train_labels.csv")
graph_df = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\graph_metrics.csv")
original = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\darknet_combined_labeled.csv", low_memory=False)
print("✅ Loaded.")

# === Clean subset
subset = original[['ip.src', 'ip.proto', 'frame.len']].copy()
subset = subset.rename(columns={'ip.src': 'node'})
subset['ip.proto'] = pd.to_numeric(subset['ip.proto'], errors='coerce').fillna(0).astype(int)
subset['frame.len'] = subset['frame.len'].astype(int)

train['ip.proto'] = train['ip.proto'].astype(int)
train['frame.len'] = train['frame.len'].astype(int)

# ✅ Filter subset to prevent memory error
print("🔍 Filtering subset before merge...")
subset['key'] = list(zip(subset['ip.proto'], subset['frame.len']))
train['key'] = list(zip(train['ip.proto'], train['frame.len']))
subset_filtered = subset[subset['key'].isin(train['key'])].drop_duplicates(subset=['ip.proto', 'frame.len'])

# Now merge
print("🔄 Merging with filtered subset...")
merged = train.merge(subset_filtered.drop(columns='key'), on=['ip.proto', 'frame.len'], how='left')
print("✅ Merge 1 done.")

# Merge with graph features
print("🔄 Merging with graph metrics...")
train_with_graph = merged.merge(graph_df, on='node', how='left')
train_with_graph = train_with_graph.dropna(subset=['betweenness', 'closeness', 'pagerank']).reset_index(drop=True)
train_labels = train_labels.loc[train_with_graph.index].reset_index(drop=True)
print("✅ Merge 2 done.")

# Train
print("🚀 Training XGBoost with graph features...")
features = ['ip.proto', 'frame.len', 'betweenness', 'closeness', 'pagerank']
model = xgb.XGBClassifier()
model.fit(train_with_graph[features], train_labels['label'])
print("✅ Training done.")

# Save
joblib.dump(model, r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\xgb_model_with_graph.pkl")
print("✅ Trained and saved xgb_model_with_graph.pkl")
