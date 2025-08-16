import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -------- Paths (edit if needed) --------
VAL_CSV    = r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val.csv"
VAL_LABELS = r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val_labels.csv"
OUT_DIR    = r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\confusion_matrices"

MODELS = {
    "Logistic Regression (Baseline)": {
        "path": r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\logistic_regression_model.pkl",
        "features": ['ip.proto', 'frame.len']
    },
    "Random Forest (Baseline)": {
        "path": r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\random_forest_model.pkl",
        "features": ['ip.proto', 'frame.len']
    },
    "XGBoost (Baseline)": {
        "path": r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\xgb_model.pkl",
        "features": ['ip.proto', 'frame.len']
    }
}

# If your y_true is numeric (0..3) and you want pretty labels, set a mapping here.
# Leave as {} to skip mapping.
CLASS_MAP = {  # example: {0:"Tor", 1:"I2P", 2:"Freenet", 3:"Zeronet"}
}

# -------- Load data --------
print("Loading validation data...")
X_val = pd.read_csv(VAL_CSV)
y_true = pd.read_csv(VAL_LABELS, usecols=['label'])['label']

# Clean / cast just in case
X_val = X_val.replace([np.inf, -np.inf], np.nan)
X_val = X_val.fillna(X_val.median(numeric_only=True))

# Auto-detect actual labels in y_true (after optional mapping)
if CLASS_MAP:
    # Map numeric -> names for y_true
    y_true_mapped = y_true.map(CLASS_MAP)
    if y_true_mapped.isna().any():
        missing = sorted(y_true[y_true_mapped.isna()].unique())
        raise ValueError(f"y_true contains values not in CLASS_MAP: {missing}")
    y_used = y_true_mapped
    display_labels = [CLASS_MAP[k] for k in sorted(CLASS_MAP.keys())]
else:
    y_used = y_true
    # Let sklearn determine labels from y_true; we’ll also display them
    display_labels = sorted(y_used.unique().tolist())

print("Detected labels in y_true:", display_labels)

os.makedirs(OUT_DIR, exist_ok=True)

for name, cfg in MODELS.items():
    print(f"\nProcessing {name} ...")
    # Load model
    clf = joblib.load(cfg["path"])
    # Predict
    X_feat = X_val[cfg["features"]]
    y_pred = clf.predict(X_feat)

    # If we mapped y_true to names, we must also map y_pred to names
    if CLASS_MAP:
        inv_map = {k: v for k, v in CLASS_MAP.items()}  # same mapping
        # If your model outputs numeric classes that are keys in CLASS_MAP, map them:
        y_pred_mapped = pd.Series(y_pred).map(inv_map)
        if y_pred_mapped.isna().any():
            missing = sorted(set(y_pred) - set(inv_map.keys()))
            raise ValueError(f"Pred contains classes not in CLASS_MAP: {missing}")
        y_plot = y_pred_mapped
    else:
        y_plot = y_pred

    # Confusion matrix using labels present in y_used (safe & consistent)
    cm = confusion_matrix(y_used, y_plot, labels=display_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
    ax.set_title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_cm.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")

    # Print classification report (matches your table metrics)
    print("\nClassification report:")
    print(classification_report(y_used, y_plot, zero_division=0))
