import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

# --- Load data ---
cols_base  = ['ip.proto', 'frame.len']
cols_graph = ['ip.proto', 'frame.len', 'betweenness', 'closeness', 'pagerank']

val_df         = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val.csv", usecols=cols_base)
val_labels     = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val_labels.csv", usecols=['label'])
val_with_graph = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val_with_graph.csv", usecols=cols_graph)

# --- Clean numeric columns: replace inf -> NaN, then fill ---
def clean_numeric(df):
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    # Option A (safe for graph features): fill NaN with 0
    df[num_cols] = df[num_cols].fillna(0)
    # cast to float32 for speed/memory
    df[num_cols] = df[num_cols].astype('float32')
    return df

val_df         = clean_numeric(val_df)
val_with_graph = clean_numeric(val_with_graph)

# --- Optional: sanity check (prints only if something remains wrong) ---
if val_with_graph.isna().any().any():
    print("Warning: NaNs still present:\n", val_with_graph.isna().sum())

# --- Load models ---
lr_baseline = joblib.load(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\logistic_regression_model.pkl")
lr_graph    = joblib.load(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\lr_model_with_graph.pkl")

# --- Predict ---
baseline_preds = lr_baseline.predict(val_df[cols_base])
graph_preds    = lr_graph.predict(val_with_graph[cols_graph])

# --- Evaluate ---
def evaluate(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'Macro Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }

baseline_metrics = evaluate(val_labels['label'], baseline_preds)
graph_metrics    = evaluate(val_labels['label'], graph_preds)

summary_df = pd.DataFrame(
    [baseline_metrics, graph_metrics],
    index=["Logistic Regression (Baseline)", "Logistic Regression (Graph Features)"]
)

print("📊 Logistic Regression Model Comparison:\n")
print(tabulate(summary_df, headers="keys", tablefmt="github"))
