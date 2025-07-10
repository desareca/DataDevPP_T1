import pandas as pd
import numpy as np
from src.config import EPS

def create_features_and_target(df, numeric_feat, categorical_feat, target_col):
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

    return df.reset_index(drop=True), features, target_col

def build_features_data(df, numeric_feat, categorical_feat, target_col='high_tip'):
    
    # Create features
    df, features, target_col = create_features_and_target(df, numeric_feat, categorical_feat, target_col)
    print(f"Create Features ->\tNum rows: {len(df)} | Size: {df.memory_usage(deep=True).sum() / 1e9} GB")
    
    return df, features, target_col

