import numpy as np
import pandas as pd

def generate_rul_targets(df, components, cycle_col='cycle', unit_col='unit'):
    """Generate RUL columns for each component based on cycle"""
    rul_df = df[[unit_col, cycle_col]].copy()
    max_cycles = rul_df.groupby(unit_col)[cycle_col].transform('max')
    for comp in components:
        rul_df[f'RUL_{comp}'] = max_cycles - rul_df[cycle_col]
    return pd.concat([df.reset_index(drop=True), rul_df[[f'RUL_{c}' for c in components]]], axis=1)

def create_sequence_dataset(df, rul_df, window_size=30, components=None, unit_col='unit'):
    """Generate sequence samples (X) and multi-target labels (y)"""
    sequence_X, sequence_y = [], []

    merged = pd.concat([df.reset_index(drop=True), rul_df[[f'RUL_{c}' for c in components]]], axis=1)
    sensor_cols = [col for col in df.columns if 'sensor' in col]

    for unit_id in merged[unit_col].unique():
        unit_df = merged[merged[unit_col] == unit_id].reset_index(drop=True)
        for i in range(len(unit_df) - window_size):
            seq = unit_df.loc[i:i+window_size-1, sensor_cols].values
            label = unit_df.loc[i+window_size-1, [f'RUL_{c}' for c in components]].values
            sequence_X.append(seq)
            sequence_y.append(label)

    return np.array(sequence_X), np.array(sequence_y)