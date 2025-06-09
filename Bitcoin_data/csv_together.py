import os
import glob
import pandas as pd

# Get all CSV files in the current folder
csv_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.csv"))

# List to hold dataframes
dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all dataframes
if dfs:
    big_df = pd.concat(dfs, ignore_index=True)
    big_df.to_csv("all_together.csv", index=False)
    print(f"Combined {len(csv_files)} files into all_together.csv")
else:
    print("No CSV files found in the folder.")