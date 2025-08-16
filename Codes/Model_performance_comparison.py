import pandas as pd
import joblib
from sklearn.metrics import classification_report

# === Load data
val = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val.csv")
val_labels = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val_labels.csv")
val_with_graph = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val_with_graph.csv")

# === Align label rows for val_with_graph
val_labels_graph = val_labels.loc[val_with_graph.index].reset_index(drop=True)

# === Load models
baseline_model = joblib.load(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\xgb_model.pkl")
graph_model = joblib.load(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\xgb_model_with_graph.pkl")

# === Define feature columns
baseline_features = ['ip.proto', 'frame.len']
graph_features = ['ip.proto', 'frame.len', 'betweenness', 'closeness', 'pagerank']

# === Make predictions
baseline_preds = baseline_model.predict(val[baseline_features])
graph_preds = graph_model.predict(val_with_graph[graph_features])

# === Get classification reports
baseline_report = classification_report(val_labels, baseline_preds, output_dict=True)
graph_report = classification_report(val_labels_graph, graph_preds, output_dict=True)

# === Create summary table
summary_df = pd.DataFrame([
    {
        "Model": "XGBoost (Baseline)",
        "Accuracy": round(baseline_report["accuracy"], 3),
        "Macro F1": round(baseline_report["macro avg"]["f1-score"], 3),
        "Macro Precision": round(baseline_report["macro avg"]["precision"], 3),
        "Macro Recall": round(baseline_report["macro avg"]["recall"], 3),
    },
    {
        "Model": "XGBoost (Graph Features)",
        "Accuracy": round(graph_report["accuracy"], 3),
        "Macro F1": round(graph_report["macro avg"]["f1-score"], 3),
        "Macro Precision": round(graph_report["macro avg"]["precision"], 3),
        "Macro Recall": round(graph_report["macro avg"]["recall"], 3),
    }
])

# === Print result as table
print("\n📊 Model Performance Comparison:")
print(summary_df.to_markdown(index=False))
