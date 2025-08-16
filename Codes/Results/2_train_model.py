import pandas as pd
import numpy as np
import os
import joblib
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


split_path = r"MRP\Dataset\split"
model_output_path = r"Downloads\MRP\models"
os.makedirs(model_output_path, exist_ok=True)


X_train = pd.read_csv(os.path.join(split_path, "train.csv"))
y_train = pd.read_csv(os.path.join(split_path, "train_labels.csv")).values.ravel()
X_val = pd.read_csv(os.path.join(split_path, "val.csv"))
y_val = pd.read_csv(os.path.join(split_path, "val_labels.csv")).values.ravel()


classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
weight_dict = dict(zip(classes, weights))
sample_weights = np.array([weight_dict[label] for label in y_train])

#Train XGBoost with sample weights
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Predict & evaluate
y_pred = model.predict(X_val)
print("Classification Report:")
print(classification_report(y_val, y_pred, digits=2))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))


joblib.dump(model, os.path.join(model_output_path, "xgb_model.pkl"))
print("Model saved as xgb_model.pkl")
