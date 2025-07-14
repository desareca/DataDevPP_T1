import pandas as pd
import numpy as np
from src.config import TIP_THRESHOLD

def load_parquet_data(filepath):
    # Carga archivo parquet desde disco o URL
    return pd.read_parquet(filepath)

def save_parquet_data(df, filepath, compresion='snappy'):
    # Guarda datos en formato Parquet

    if isinstance(df, dict):
        df = pd.DataFrame(df)
    df.to_parquet(filepath, compression=compresion, index=False)
    print(f"Datos guardados en {filepath} con compresión {compresion}")

def basic_cleaning(df, target_col):
    # Basic cleaning
    df = df[df['fare_amount'] > 0].reset_index(drop=True)  # avoid divide-by-zero
    # add target
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[target_col] = df['tip_fraction'] > TIP_THRESHOLD
    return df

def preprocess_data(filepath, save_file_name, target_col='high_tip', verbose=False):
    """Complete pipeline: load -> clean -> features"""
    # Load
    df = load_parquet_data(filepath)
    filepath_raw = "../data/raw/"
    save_parquet_data(df, filepath_raw + save_file_name + '.parquet')

    if verbose:
        print(f"Load ->\t\t\tNum rows: {len(df)} | Size: {df.memory_usage(deep=True).sum() / 1e9} GB")

    # Clean and create target
    df = basic_cleaning(df, target_col)
    filepath_clean = "../data/interim/"
    save_parquet_data(df, filepath_clean + save_file_name + '_clean.parquet')

    
    if verbose:
        print(f"Clean ->\t\tNum rows: {len(df)} | Size: {df.memory_usage(deep=True).sum() / 1e9} GB")
    
    return df

