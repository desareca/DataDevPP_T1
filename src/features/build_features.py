import pandas as pd
import numpy as np
from src.config import EPS
from src.data.dataset import save_parquet_data


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

def build_features_data(df, numeric_feat, categorical_feat, save_file_name, target_col='high_tip', verbose=False):
    
    # Create features
    df, features, target_col = create_features_and_target(df, numeric_feat, categorical_feat, target_col)
    filepath_processed = "../data/processed/"
    save_parquet_data(df, filepath_processed + save_file_name + '_processed.parquet')
    
    if verbose:
        print(f"Create Features ->\tNum rows: {len(df)} | Size: {df.memory_usage(deep=True).sum() / 1e9} GB")
    
    return df, features, target_col


def detect_data_drift(df1, df2, columns, categorical_threshold=2, p_value_th=0.01):
    from scipy.stats import ks_2samp
    from scipy.stats import chi2_contingency
    """
    Detecta drift usando KS o Chi2 según tipo de variable
    """
    results = []
   
    for col in columns:
        data1 = df1[col].dropna()
        data2 = df2[col].dropna()
        
        # Decidir test según número de valores únicos
        unique_vals = len(set(data1.unique()) | set(data2.unique()))
        
        if unique_vals <= categorical_threshold:
                # Test Chi-cuadrado para categóricas
                counts1 = data1.value_counts()
                counts2 = data2.value_counts()
                all_cats = set(counts1.index) | set(counts2.index)

                table = [[counts1.get(cat, 0) for cat in all_cats],
                        [counts2.get(cat, 0) for cat in all_cats]]

                chi2_stat, p_value, _, _ = chi2_contingency(table)
                mean_diff = abs(data1.mean() - data2.mean()) / (data1.mean() + EPS)
                std_diff = abs(data1.std() - data2.std()) / (data1.std() + EPS)


                results.append({
                    'variable': col,
                    'test_used': 'chi2',
                    'statistic': chi2_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < p_value_th,
                    'mean_diff_pct': mean_diff * 100,
                    'std_diff_pct': std_diff * 100,
                    'unique_values': unique_vals,
                    'rel_sample_size': len(data2) / len(data1)
                })
        else:
                # Test KS para continuas
                ks_stat, p_value = ks_2samp(data1, data2)
                mean_diff = abs(data1.mean() - data2.mean()) / (data1.mean() + EPS)
                std_diff = abs(data1.std() - data2.std()) / (data1.std() + EPS)

                results.append({
                    'variable': col,
                    'test_used': 'ks',
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < p_value_th,
                    'mean_diff_pct': mean_diff * 100,
                    'std_diff_pct': std_diff * 100,
                    'unique_values': unique_vals,
                    'rel_sample_size': len(data2) / len(data1)
                })
   
    return pd.DataFrame(results)

def compare_data_drift_months(data_dict, columns, reference_month):
    """
    Compara todos los meses contra un mes de referencia
    """
    all_results = {}
    
    for month, df in data_dict.items():
        if month != reference_month:
            drift_df = detect_data_drift(data_dict[reference_month], df, columns)
            all_results[f"{month}"] = drift_df
            
            # Resumen
            drift_vars = drift_df['drift_detected'].sum()
            total_vars = len(drift_df)
            print(f"{reference_month} vs {month}: {drift_vars}/{total_vars} variables con drift")
    
    return all_results
