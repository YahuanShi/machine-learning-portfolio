import pandas as pd

def add_rolling_features(df, sensor_cols, window=5):
    for col in sensor_cols:
        df[f'{col}_mean'] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{col}_std'] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).std())
        df[f'{col}_diff'] = df.groupby('unit')[col].diff()
    return df

def generate_label(df, threshold=30):
    df['label'] = (df['RUL'] <= threshold).astype(int)
    return df