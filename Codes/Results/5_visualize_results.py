import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


split_path = r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split"
models = ["xgboost", "randomforest", "logisticregression"]
model_names = {
    "xgboost": "XGBoost",
    "randomforest": "Random Forest",
    "logisticregression": "Logistic Regression"
}

results = []

for model in models:
    json_path = os.path.join(split_path, f"{model}_classification_report.json")
    with open(json_path, "r") as f:
        report = json.load(f)

    for cls in report:
        if cls in ["accuracy", "macro avg", "weighted avg"]:
            continue  
        results.append({
            "Model": model_names[model],
            "Class": cls,
            "Precision": report[cls]["precision"],
            "Recall": report[cls]["recall"],
            "F1-Score": report[cls]["f1-score"]
        })


df = pd.DataFrame(results)

# Sort classes numerically if possible
df["Class"] = df["Class"].astype(str)
try:
    df["Class"] = df["Class"].astype(int).astype(str)
except:
    pass


melted = df.melt(id_vars=["Model", "Class"], value_vars=["Precision", "Recall", "F1-Score"],
                 var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 6))
sns.barplot(data=melted, x="Class", y="Score", hue="Model", ci=None)
plt.title("Model Comparison per Class (Precision, Recall, F1-Score)")
plt.ylim(0, 1)
plt.xlabel("Class")
plt.ylabel("Score")
plt.xticks(ticks=[0, 1, 2, 3], labels=['Tor', 'I2P', 'Freenet', 'Zeronet'])
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(os.path.join(split_path, "model_comparison_per_class.png"))
plt.show()


metrics = ["Precision", "Recall", "F1-Score"]

for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Class", y=metric, hue="Model", palette="Set2")
    plt.title(f"{metric} per Class by Model")
    plt.ylim(0, 1)
    plt.xlabel("Class")
    plt.ylabel(metric)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['Tor', 'I2P', 'Freenet', 'Zeronet'])
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(split_path, f"{metric.lower()}_per_class.png"))
    plt.show()
