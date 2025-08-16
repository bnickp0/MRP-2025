import pandas as pd
import joblib
from sklearn.metrics import classification_report
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("✅ Starting Random Forest model evaluation script...")

# 📥 Load validation sets
print("📂 Loading validation datasets...")
val_df = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val.csv")
print("   - val.csv loaded with shape:", val_df.shape)

val_labels = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val_labels.csv")
print("   - val_labels.csv loaded with shape:", val_labels.shape)

val_with_graph = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val_with_graph.csv")
print("   - val_with_graph.csv loaded with shape:", val_with_graph.shape)

# 📥 Load models
print("📂 Loading Random Forest models...")
rf_baseline = joblib.load(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\random_forest_model.pkl")
print("   - Random Forest Baseline model loaded.")

rf_graph = joblib.load(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\rf_model_with_graph.pkl")
print("   - Random Forest Graph model loaded.")

# ✅ Predict
print("🤖 Running predictions...")
baseline_preds = rf_baseline.predict(val_df[['ip.proto', 'frame.len']])
print("   - Baseline predictions done.")

graph_preds = rf_graph.predict(val_with_graph[['ip.proto', 'frame.len', 'betweenness', 'closeness', 'pagerank']])
print("   - Graph predictions done.")

# 🧮 Evaluation
def evaluate(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'Macro Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }

print("📏 Calculating metrics...")
baseline_metrics = evaluate(val_labels['label'], baseline_preds)
print("   - Baseline metrics calculated.")

graph_metrics = evaluate(val_labels['label'], graph_preds)
print("   - Graph metrics calculated.")

# 📝 Show as table
summary_df = pd.DataFrame([baseline_metrics, graph_metrics], index=[
    "Random Forest (Baseline)",
    "Random Forest (Graph Features)"
])

print("\n📊 Random Forest Model Comparison:\n")
print(tabulate(summary_df, headers='keys', tablefmt='github'))

print("\n✅ Script finished successfully.")
