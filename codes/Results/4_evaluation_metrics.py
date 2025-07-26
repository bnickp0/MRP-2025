import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import json


split_path = "Downloads\MRP\Dataset\split"
models = ["xgboost", "randomforest", "logisticregression"]
model_names = {
    "xgboost": "XGBoost",
    "randomforest": "Random Forest",
    "logisticregression": "Logistic Regression"
}


y_val = pd.read_csv(os.path.join(split_path, "val_labels.csv")).values.ravel()


for model in models:
    print(f"\n Evaluating {model_names[model]}...")

   
    pred_path = os.path.join(split_path, f"{model}_predictions.csv")
    df = pd.read_csv(pred_path)
    y_pred = df["predicted"].values
    y_true = df["actual"].values[:len(y_pred)]

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=2, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, digits=2, zero_division=0)

    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(report_text)

    with open(os.path.join(split_path, f"{model}_report.txt"), "w") as f:
        f.write(f"{model_names[model]} Classification Report\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n\n")
        f.write(report_text)

  
    with open(os.path.join(split_path, f"{model}_classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Confusion Matrix (raw)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_names[model]} Confusion Matrix (Raw)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(split_path, f"{model}_confusion_matrix.png"))
    plt.close()

    # Confusion Matrix (normalized)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    labels = np.unique(np.concatenate((y_true, y_pred)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_names[model]} Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(split_path, f"{model}_confusion_matrix_normalized.png"))
    plt.close()

    print(f"Saved report and confusion matrices for {model_names[model]}")

# NOTE: KNN was excluded from final evaluation due to computational inefficiency on large datasets.
