import pandas as pd

def load_fd001(filepath):
    df = pd.read_csv(filepath, sep=' ', header=None)
    df.drop(columns=[26, 27], inplace=True)
    df.columns = ['unit','cycle','op1','op2','op3'] + [f'sensor{i}' for i in range(1,22)]
    return df

def add_rul(df):
    df['RUL'] = df.groupby('unit')['cycle'].transform('max') - df['cycle']
    return df