
import pandas as pd
import os

input_folder = r"Dataset"
output_folder = r"labeled_output"
label = "Freenet"  # Change this per network (Tor, I2P, etc)
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_folder, file))
        df['label'] = label
        df.to_csv(os.path.join(output_folder, f"{label.lower()}_labeled.csv"), index=False)
