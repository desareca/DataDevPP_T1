import pandas as pd
import numpy as np

def load_parquet_data(filepath):
    #Carga archivo parquet desde disco o URL
    return pd.read_parquet(filepath)

def basic_cleaning(df, target_col):
    # Basic cleaning
    df = df[df['fare_amount'] > 0].reset_index(drop=True)  # avoid divide-by-zero
    # add target
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[target_col] = df['tip_fraction'] > 0.2
    return df

def create_features_and_target(df, numeric_feat, categorical_feat, target_col, EPS=1e-7):
    # add features
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
    df['work_hours'] = (df['pickup_weekday'] >= 0) & (df['pickup_weekday'] <= 4) & (df['pickup_hour'] >= 8) & (df['pickup_hour'] <= 18)
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.seconds
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + EPS)

    # drop unused columns
    features = numeric_feat + categorical_feat
    df = df[['tpep_dropoff_datetime'] + features + [target_col]]
    df.loc[:,features + [target_col]] = df[features + [target_col]].astype("float32").fillna(-1.0)

    # convert target to int32 for efficiency (it's just 0s and 1s)
    df.loc[:,target_col] = df[target_col].astype("int32")

    return df.reset_index(drop=True)

def process_data(filepath, numeric_feat, categorical_feat, target_col='high_tip'):
    """Complete pipeline: load -> clean -> features"""
    # Load
    df = load_parquet_data(filepath)
    print(f"Loaded: {df.shape}")

    # Clean and create target
    df = basic_cleaning(df, target_col)
    
    # Create features
    df = create_features_and_target(df, numeric_feat, categorical_feat, target_col)
    
    return df


