import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# ---- 1) Fast, lean loading ----
usecols = ['ip.proto', 'frame.len']  # only what we need
dtypes = {'ip.proto': 'int16', 'frame.len': 'int32'}  # compact ints

train = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\train.csv",
                    usecols=usecols, dtype=dtypes)
train_labels = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\train_labels.csv",
                           usecols=['label'])

# ---- 2) Convert once to NumPy float32 ----
X = train[['ip.proto', 'frame.len']].to_numpy(dtype=np.float32, copy=False)
y = train_labels['label'].to_numpy()

# Optional: quick sanity cleanup (cheap)
mask = np.isfinite(X).all(axis=1)
if mask.sum() != len(X):
    X, y = X[mask], y[mask]

print("🚀 Training baseline Logistic Regression (fast settings)...")

# ---- 3) Use a solver that scales on many rows ----
# saga + L2 works well for large datasets; higher tol and smaller C speed up convergence.
lr_baseline_model = LogisticRegression(
    solver='saga',          # good for large n_samples
    penalty='l2',
    C=0.5,                  # stronger regularization => fewer iterations
    tol=1e-2,               # looser tolerance => earlier stop
    max_iter=200,           # usually enough with 2 features
    n_jobs=-1,              # use all cores where supported
    verbose=1,
    random_state=42
)

lr_baseline_model.fit(X, y)
print("✅ Baseline LR trained.")

# ---- 4) Save (compressed) ----
out_path = r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\split\logistic_regression_model.pkl"
joblib.dump(lr_baseline_model, out_path, compress=('xz', 3))
print(f"✅ Saved: {out_path}")
