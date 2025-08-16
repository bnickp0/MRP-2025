import pandas as pd

# Load your dataset (update the path to your dataset CSV)
df = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\darknet_combined_labeled.csv")

# Show all field names (columns)
print(df.columns.tolist())

# Optional: show first few rows to see data
print(df.head())
