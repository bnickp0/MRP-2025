import pandas as pd
import os
import time
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


split_path = "MRP\Dataset\split"
X_train = pd.read_csv(os.path.join(split_path, "train.csv"))
y_train = pd.read_csv(os.path.join(split_path, "train_labels.csv")).values.ravel()
X_val = pd.read_csv(os.path.join(split_path, "val.csv"))
y_val = pd.read_csv(os.path.join(split_path, "val_labels.csv")).values.ravel()


classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
weight_dict = dict(zip(classes, weights))
sample_weights = np.array([weight_dict[label] for label in y_train])

# Define models to compare
models = {
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    "randomforest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "logisticregression": LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", multi_class="multinomial")
}


output_dir = split_path
os.makedirs(output_dir, exist_ok=True)

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🔄 Training {name.capitalize()}...")
    start = time.time()

    if name == "xgboost":
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    duration = time.time() - start

    print(f"\n⏱{name.capitalize()} trained in {duration:.2f} seconds")
    print(f"\n {name.capitalize()} Classification Report:\n")
    print(classification_report(y_val, y_pred, digits=2))

   
    pd.DataFrame({
        "actual": y_val,
        "predicted": y_pred
    }).to_csv(os.path.join(output_dir, f"{name}_predictions.csv"), index=False)

    print(f" Saved {name} predictions to {name}_predictions.csv")

# NOTE: KNN was excluded from final results due to extremely long runtime on the full dataset.
# It was tested separately on a smaller subset and found to be computationally infeasible.
