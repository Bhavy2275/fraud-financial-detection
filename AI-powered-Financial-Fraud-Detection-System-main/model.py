import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
import pickle

def engineer_features(df):
    """Create advanced features for better fraud detection"""
    # Time-based features
    df['DATE'] = pd.to_datetime(df['DATE'], format='mixed')
    df['VALUE DATE'] = pd.to_datetime(df['VALUE DATE'], format='mixed')
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['hour_of_day'] = df['DATE'].dt.hour
    df['month'] = df['DATE'].dt.month
    
    # Handle infinite and null values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Transaction amount features (with safety checks)
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    df['transaction_amount'] = df['transaction_amount'].fillna(0)
    df['transaction_amount_log'] = np.log1p(df['transaction_amount'].clip(lower=0))
    
    # Safe division with error handling
    df['withdrawal_to_balance_ratio'] = (
        df['WITHDRAWAL AMT'].fillna(0) / 
        (df['BALANCE AMT'].fillna(0) + 1)
    ).clip(lower=0, upper=1000)
    
    df['deposit_to_balance_ratio'] = (
        df['DEPOSIT AMT'].fillna(0) / 
        (df['BALANCE AMT'].fillna(0) + 1)
    ).clip(lower=0, upper=1000)
    
    # Rolling window features with error handling
    df['avg_transaction_amount'] = (
        df.groupby('Account No')['transaction_amount']
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        .fillna(0)
    )
    
    df['transaction_amount_diff'] = (
        df.groupby('Account No')['transaction_amount']
        .diff()
        .fillna(0)
    )
    
    # Frequency features
    df['daily_transaction_count'] = (
        df.groupby(['Account No', df['DATE'].dt.date])
        .cumcount()
        .fillna(0)
    )
    
    # Replace any remaining infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def create_anomaly_labels(df):
    """Create sophisticated anomaly detection labels"""
    # Ensure all values are finite before calculating quantiles
    numeric_cols = ['transaction_amount', 'daily_transaction_count', 
                   'transaction_amount_diff', 'withdrawal_to_balance_ratio']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    conditions = [
        # High amount transactions
        (df['transaction_amount'] > df['transaction_amount'].quantile(0.95)),
        # Unusual transaction patterns
        (df['daily_transaction_count'] > df['daily_transaction_count'].quantile(0.95)),
        # Large changes in transaction amounts
        (abs(df['transaction_amount_diff']) > df['transaction_amount'].quantile(0.95)),
        # Unusual withdrawal patterns
        (df['withdrawal_to_balance_ratio'] > df['withdrawal_to_balance_ratio'].quantile(0.95))
    ]
    
    df['is_suspicious'] = np.where(np.any(conditions, axis=0), 1, 0)
    return df

def train_model():
    print("Loading dataset...")
    df = pd.read_csv('transactions.csv')
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Creating anomaly labels...")
    df = create_anomaly_labels(df)
    
    # Adjust class weights to heavily penalize false positives
    class_weights = {0: 1, 1: 2}  # Penalize false positives more
    
    # Create a more conservative model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        scale_pos_weight=10,  # Handle class imbalance
        min_child_weight=5,   # More conservative predictions
        subsample=0.8,        # Reduce overfitting
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Prepare features and target
    features = [
        'WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT', 'transaction_amount',
        'day_of_week', 'hour_of_day', 'month',
        'transaction_amount_log', 'withdrawal_to_balance_ratio', 
        'deposit_to_balance_ratio', 'avg_transaction_amount',
        'transaction_amount_diff', 'daily_transaction_count'
    ]
    X = df[features].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    y = df['is_suspicious']
    
    # Split with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate with focus on false positives
    y_pred = model.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate false positive rate
    tn, fp, fn, tp = conf_matrix.ravel()
    fpr = fp / (fp + tn)
    
    print("\nModel Performance:")
    print(f"False Positive Rate: {fpr:.4%}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Adjust probability threshold if needed
    if fpr > 0.001:  # If false positive rate is above 0.1%
        print("\nAdjusting probability threshold to reduce false positives...")
        # Find threshold that gives desired false positive rate
        probs = model.predict_proba(X_test_scaled)[:, 1]
        thresholds = np.arange(0.5, 0.99, 0.01)
        for threshold in thresholds:
            y_pred_adj = (probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_adj).ravel()
            fpr_adj = fp / (fp + tn)
            if fpr_adj <= 0.001:
                print(f"New threshold: {threshold:.2f}")
                print(f"New false positive rate: {fpr_adj:.4%}")
                break
    
    # Save the model and scaler
    print("\nSaving model and scaler...")
    with open('fraud_detection_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model training completed!")
    return model, scaler

if __name__ == "__main__":
    train_model()
