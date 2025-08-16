import os, sys, time
print("1) Script start", flush=True)

# ========= CONFIG (edit these paths as needed) =========
MODEL_PATH = r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\random_forest_model.pkl"  # <- your RF baseline file
VAL_CSV    = r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val.csv"
VAL_LABELS = r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\val_labels.csv"
OUT_DIR    = r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\shap_outputs"

FEATURES   = ['ip.proto', 'frame.len']   # RF Baseline features
N_SAMPLE   = 5000                         # set to None to use ALL rows
# =======================================================

print("2) Checking files exist...", flush=True)
for p in [MODEL_PATH, VAL_CSV, VAL_LABELS]:
    if not os.path.isfile(p):
        print(f"   ERROR: missing file -> {p}", flush=True)
        sys.exit(1)
print("   OK: all required files found.", flush=True)

print("3) Importing libraries & setting headless backend...", flush=True)
import matplotlib
matplotlib.use("Agg")   # avoid GUI blocking
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

try:
    import shap
    print("   OK: shap imported (v", getattr(shap, "__version__", "unknown"), ")", flush=True)
except Exception as e:
    print("   ERROR importing shap:", repr(e), flush=True)
    sys.exit(1)

t0 = time.time()

print("4) Loading validation data...", flush=True)
X = pd.read_csv(VAL_CSV, usecols=FEATURES)
y = pd.read_csv(VAL_LABELS, usecols=['label'])['label']
print(f"   Shapes -> X:{X.shape}  y:{y.shape}", flush=True)

if N_SAMPLE is not None and len(X) > N_SAMPLE:
    X = X.iloc[:N_SAMPLE].copy()
    y = y.iloc[:N_SAMPLE].copy()
    print(f"   Sampled first {N_SAMPLE} rows -> X:{X.shape}  y:{y.shape}", flush=True)

print("5) Cleaning NaN/Inf...", flush=True)
num = X.select_dtypes(include='number').columns
bad = X[num].replace([np.inf, -np.inf], np.nan).isna().sum()
if bad.sum():
    print("   Found NaNs/Inf:", bad[bad>0].to_dict(), flush=True)
X[num] = X[num].replace([np.inf, -np.inf], np.nan)
X[num] = X[num].fillna(X[num].median()).astype('float32')
print("   OK: cleaned.", flush=True)

print("6) Loading RandomForest model...", flush=True)
try:
    rf = joblib.load(MODEL_PATH)
    print("   OK: model loaded:", type(rf).__name__, flush=True)
except Exception as e:
    print("   ERROR loading model:", repr(e), flush=True)
    sys.exit(1)

print("7) Building TreeExplainer...", flush=True)
try:
    explainer = shap.TreeExplainer(rf)
    print("   OK: explainer ready.", flush=True)
except Exception as e:
    print("   ERROR creating TreeExplainer:", repr(e), flush=True)
    sys.exit(1)

print("8) Computing SHAP values (may take a bit)...", flush=True)
try:
    shap_values = explainer.shap_values(X)
    # normalize to (classes, samples, features)
    def to_csf(sv):
        if isinstance(sv, list):
            arr = np.stack([np.asarray(m) for m in sv], axis=0)  # (C, S, F)
            return arr
        arr = np.asarray(sv)
        if arr.ndim == 4 and arr.shape[0] == 1:  # (1,S,F,C)
            arr = arr[0]
        if arr.ndim == 3:                         # (S,F,C) or (C,S,F)
            if arr.shape[1] == len(FEATURES):     # (S,F,C)
                arr = np.moveaxis(arr, 2, 0)      # -> (C,S,F)
        return arr
    csf = to_csf(shap_values)
    print(f"   SHAP shape normalized to (C,S,F): {csf.shape}", flush=True)
except Exception as e:
    print("   ERROR computing SHAP values:", repr(e), flush=True)
    sys.exit(1)

print("9) Aggregating mean |SHAP| per feature...", flush=True)
mean_abs = np.mean(np.abs(csf), axis=(0, 1))  # -> (F,)
imp_df = pd.DataFrame({"Feature": FEATURES, "Mean|SHAP|": mean_abs})\
         .sort_values("Mean|SHAP|", ascending=False)
print("\nTop features by mean |SHAP|:\n", imp_df.to_string(index=False), flush=True)

print("10) Saving outputs...", flush=True)
os.makedirs(OUT_DIR, exist_ok=True)
imp_csv = os.path.join(OUT_DIR, "rf_baseline_shap_importance.csv")
imp_df.to_csv(imp_csv, index=False)

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "rf_baseline_shap_summary_bar.png")
plt.savefig(bar_path, dpi=200)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
dot_path = os.path.join(OUT_DIR, "rf_baseline_shap_summary_dot.png")
plt.savefig(dot_path, dpi=200)
plt.close()

elapsed = time.time() - t0
print(f"\n11) Done. Saved files:\n - {imp_csv}\n - {bar_path}\n - {dot_path}", flush=True)
print(f"Total time: {elapsed:.2f}s", flush=True)
