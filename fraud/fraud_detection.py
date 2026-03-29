"""
Fraud Anomaly Detection Module

This module implements anomaly detection using DBSCAN clustering
to identify fraudulent transactions in financial data.
Loads transaction data from CSV file.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def create_fraud_csv():
    """
    Create and save sample transaction dataset to CSV file.
    
    Returns:
        str: Path to the created CSV file
    """
    np.random.seed(42)
    
    # Create normal transactions
    n_normal = 95
    normal_amounts = np.random.uniform(50, 5000, n_normal)
    normal_times = np.random.uniform(6, 23, n_normal)  # 6am to 11pm
    normal_account_age = np.random.uniform(100, 3650, n_normal)  # 100 to 10 years
    
    # Create anomalous (fraudulent) transactions
    n_anomaly = 5
    anomaly_amounts = np.random.uniform(15000, 50000, n_anomaly)  # Very large amounts
    anomaly_times = np.random.uniform(0, 5, n_anomaly)  # Unusual early morning times
    anomaly_account_age = np.random.uniform(1, 50, n_anomaly)  # Very new accounts
    
    # Combine normal and anomalous data
    amounts = np.concatenate([normal_amounts, anomaly_amounts])
    times = np.concatenate([normal_times, anomaly_times])
    account_ages = np.concatenate([normal_account_age, anomaly_account_age])
    
    # Create DataFrame
    data = {
        'transaction_id': range(1, len(amounts) + 1),
        'transaction_amount': amounts,
        'transaction_time': times,
        'account_age_days': account_ages
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    csv_path = Path(__file__).parent.parent / 'data' / 'transactions_fraud.csv'
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f"✓ CSV file created: {csv_path}")
    return csv_path


def load_fraud_dataset(csv_path):
    """
    Load transaction dataset from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Transaction data loaded from CSV
    """
    df = pd.read_csv(csv_path)
    return df


def apply_standardscaler(features):
    """
    Apply StandardScaler to normalize features.
    
    Args:
        features: Raw feature data
        
    Returns:
        tuple: Scaled features and the scaler object
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


def train_dbscan_model(scaled_features, eps=0.5, min_samples=5):
    """
    Train DBSCAN clustering model for anomaly detection.
    
    Args:
        scaled_features: Standardized feature data
        eps: Maximum distance between points in a cluster
        min_samples: Minimum number of samples to form a dense region
        
    Returns:
        DBSCAN: Fitted DBSCAN model
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_features)
    return dbscan, clusters


def run_fraud_detection(verbose=True):
    """
    Main function to run the complete fraud detection pipeline.
    """
    if verbose:
        print("\n" + "#"*60)
        print("# FRAUD ANOMALY DETECTION MODULE")
        print("#"*60)
    
    # Create CSV file if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    csv_path = data_dir / 'transactions_fraud.csv'
    
    if not csv_path.exists():
        create_fraud_csv()
    
    # Load dataset from CSV
    if verbose:
        print(f"\n📂 Loading data from: {csv_path}")
    df = load_fraud_dataset(csv_path)
    if verbose:
        print(f"✓ Dataset loaded with {len(df)} transactions")
    
        # Display first few rows
        print(f"\nFirst 5 records:")
        print(df.head().to_string(index=False))
    
        # Display dataset statistics
        print("\nDataset Statistics:")
        print(f"  Transaction Amount - Min: ${df['transaction_amount'].min():.2f}, "
              f"Max: ${df['transaction_amount'].max():.2f}, "
              f"Mean: ${df['transaction_amount'].mean():.2f}")
        print(f"  Transaction Time - Min: {df['transaction_time'].min():.1f}h, "
              f"Max: {df['transaction_time'].max():.1f}h")
        print(f"  Account Age - Min: {df['account_age_days'].min():.1f} days, "
              f"Max: ${df['account_age_days'].max():.1f} days")
    
    # Extract features
    features = df[['transaction_amount', 'transaction_time', 'account_age_days']].values
    
    # Apply StandardScaler normalization
    if verbose:
        print("\n" + "="*60)
        print("APPLYING STANDARDSCALER NORMALIZATION")
        print("="*60)
    scaled_features, scaler = apply_standardscaler(features)
    if verbose:
        print("✓ Features scaled successfully")
        print(f"✓ Scaled data shape: {scaled_features.shape}")
    
    # Train DBSCAN model
    if verbose:
        print("\n" + "="*60)
        print("TRAINING DBSCAN CLUSTERING MODEL")
        print("="*60)
    dbscan, clusters = train_dbscan_model(scaled_features, eps=0.8, min_samples=3)
    
    # Add cluster assignments to dataframe
    df['cluster'] = clusters
    
    # Identify outliers (cluster == -1 indicates anomalies)
    outliers = df[df['cluster'] == -1]
    normal_transactions = df[df['cluster'] != -1]
    
    if verbose:
        print(f"✓ Clustering Results:")
        print(f"  Normal transaction clusters: {len(normal_transactions)}")
        print(f"  Anomalies detected (cluster = -1): {len(outliers)}")
    
        # Display fraud detection results
        print("\n" + "="*60)
        print("FRAUD DETECTION RESULTS")
        print("="*60)
    
        print(f"\nNormal Transactions (First 5):")
        print(normal_transactions[['transaction_id', 'transaction_amount', 
                                   'transaction_time', 'account_age_days', 'cluster']].head().to_string(index=False))
    
    if len(outliers) > 0:
        if verbose:
            print(f"\nFraudulent Transactions (Outliers - Cluster = -1):")
            print(outliers[['transaction_id', 'transaction_amount', 
                           'transaction_time', 'account_age_days', 'cluster']].to_string(index=False))
        
            print(f"\nFraud Summary:")
            for idx, row in outliers.iterrows():
                print(f"\n  ⚠️  FRAUD ALERT - Transaction ID: {int(row['transaction_id'])}")
                print(f"      Amount: ${row['transaction_amount']:.2f}")
                print(f"      Time: {row['transaction_time']:.1f}:00 hours")
                print(f"      Account Age: {row['account_age_days']:.1f} days")
    else:
        if verbose:
            print("\nNo fraudulent transactions detected.")
    
    return df, outliers
