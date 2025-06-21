
import pandas as pd
import glob

files = glob.glob("C:\\path\\to\\labeled_output\\*_labeled.csv")
df_combined = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
df_combined.to_csv("darknet_combined_labeled.csv", index=False)
