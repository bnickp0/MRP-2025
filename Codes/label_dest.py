import pandas as pd

# Load your dataset
df = pd.read_csv(r"C:\Users\bpanjehpour\Downloads\MRP\Dataset\darknet_combined_labeled.csv")

# Assuming the column with labels is called 'label' or similar (adjust if needed)
label_distribution = df['label'].value_counts(normalize=True) * 100  # Converts to percentages
label_distribution = label_distribution.round(2)  # Round for clean output

print(label_distribution)
