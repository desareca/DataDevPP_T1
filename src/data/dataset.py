import pandas as pd
import numpy as np
from src.config import TIP_THRESHOLD

def load_parquet_data(filepath):
    #Carga archivo parquet desde disco o URL
    return pd.read_parquet(filepath)

def basic_cleaning(df, target_col):
    # Basic cleaning
    df = df[df['fare_amount'] > 0].reset_index(drop=True)  # avoid divide-by-zero
    # add target
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[target_col] = df['tip_fraction'] > TIP_THRESHOLD
    return df

def preprocess_data(filepath, target_col='high_tip'):
    """Complete pipeline: load -> clean -> features"""
    # Load
    df = load_parquet_data(filepath)
    print(f"Load ->\t\tNum rows: {len(df)} | Size: {df.memory_usage(deep=True).sum() / 1e9} GB")

    # Clean and create target
    df = basic_cleaning(df, target_col)
    print(f"Clean ->\tNum rows: {len(df)} | Size: {df.memory_usage(deep=True).sum() / 1e9} GB")
    
    return df

