
import pandas as pd
import numpy as np
import scorecardpy as sc
import os
import json

def smart_fillna(df, method='median'):
    """Replace -999 or NaN with median or zero."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0 or (df[col] == -999).sum() > 0:
            df[col] = df[col].replace(-999, np.nan)
            if method == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif method == 'zero':
                df[col] = df[col].fillna(0)
    return df

def apply_woe_binning(df, target='SeriousDlqin2yrs'):
    """Apply WOE binning and return bins and WOE transformed df."""
    bins = sc.woebin(df, y=target)
    woe_df = sc.woebin_ply(df, bins)
    return bins, woe_df

def select_variables_by_iv(bins, threshold=0.02):
    """Select variables with IV > threshold, return _woe column names."""
    iv_list = []
    for var in bins.keys():
        iv_val = bins[var]['total_iv'].values[0]
        iv_list.append({'variable': var, 'info_value': iv_val})
    iv_df = pd.DataFrame(iv_list)
    selected_vars = iv_df[iv_df['info_value'] > threshold]['variable'].tolist()
    return [f"{var}_woe" for var in selected_vars]

def save_woe_data(woe_df, bins, output_dir='../outputs'):
    """Save WOE dataframe and bin definitions."""
    os.makedirs(output_dir, exist_ok=True)
    woe_df.to_csv(os.path.join(output_dir, 'woe_train_data.csv'), index=False)
    bins_dict = {k: v.to_dict(orient="list") for k, v in bins.items()}
    with open(os.path.join(output_dir, 'woe_bins.json'), 'w') as f:
        json.dump(bins_dict, f)
