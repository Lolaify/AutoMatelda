# This script adds synthetic semantic comain errors to a dateset
# by replacing random cell values with values somewhere else in the dataset

import copy
import math
import os
import pandas as pd
import random

dgov_typo_path = "/home/micro/Documents/Code/AutoMatelda/datasets/DGov_Typo"
save_to = "/home/micro/Documents/Code/AutoMatelda/datasets/DGov_SO"

random.seed(42)

def get_random_cell():
    cell_id = random.randint(0, n_cells-1)
    for table, df in clean_datasets.items():
        if cell_id >= dataset_sizes[table]:
            cell_id -= dataset_sizes[table]
            continue
        num_cols = df.shape[1]
        row = cell_id // num_cols
        col = cell_id % num_cols

        return table, row, col
def get_random_value(excluded_values = set()):
    table, row, col = get_random_cell()
    value = clean_datasets[table].iat[row, col]
    while value in excluded_values:
        table, row, col = get_random_cell()
        value = clean_datasets[table].iat[row, col]
    return clean_datasets[table].iat[row, col]

# --- Load Clean Dataset ---
clean_datasets = {}
dataset_sizes = {}
n_cells = 0
for table in sorted(os.listdir(dgov_typo_path)):
    clean_datasets[table] = pd.read_csv(os.path.join(dgov_typo_path, table, 'clean.csv'), dtype=str)
    dataset_sizes[table] = clean_datasets[table].shape[0] * clean_datasets[table].shape[1]
    n_cells += dataset_sizes[table]

# --- Add Synthetic Errors to Dataset ---
dirty_datasets = copy.deepcopy(clean_datasets)
num_errors = math.floor(n_cells * 0.05)
introduced_errors = []
for _ in range(0, num_errors):
    table, row, col = get_random_cell()
    while (table, row, col) in introduced_errors:
        table, row, col = get_random_cell()
    introduced_errors.append((table, row, col))
    df = clean_datasets[table]
    excluded_values = set(df[df.columns[col]].values)
    dirty_datasets[table].iat[row, col] = get_random_value(excluded_values)
print(f"introduced {num_errors} errors")

# --- Save Dataset ---
print(f"Saving Dataset to {save_to}")
for table in clean_datasets.keys():
    dirty_df = pd.DataFrame(dirty_datasets[table])
    clean_df = pd.DataFrame(clean_datasets[table])
    os.makedirs(os.path.join(save_to, table), exist_ok=True)
    dirty_df.to_csv(os.path.join(save_to, table, 'dirty.csv'), index=False)
    clean_df.to_csv(os.path.join(save_to, table, 'clean.csv'), index=False)