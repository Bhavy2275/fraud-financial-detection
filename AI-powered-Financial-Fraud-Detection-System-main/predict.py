import pandas as pd
import pickle
from model import engineer_features

def load_model_and_scaler():
    with open('fraud_detection_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_transaction(transaction_data, model, scaler, features):
    """
    Predict if a single transaction is fraudulent
    """
    # Convert single transaction to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Engineer features
    df = engineer_features(df)
    
    # Select and scale features
    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Make prediction
    fraud_probability = model.predict_proba(X_scaled)[0][1]
    is_fraud = model.predict(X_scaled)[0]
    
    return {
        'is_fraud': bool(is_fraud),
        'fraud_probability': float(fraud_probability),
        'risk_level': 'High' if fraud_probability > 0.7 else 'Medium' if fraud_probability > 0.3 else 'Low'
    }

def get_feature_names():
    return [
        'WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT', 'transaction_amount',
        'day_of_week', 'hour_of_day', 'month',
        'transaction_amount_log', 'withdrawal_to_balance_ratio', 
        'deposit_to_balance_ratio', 'avg_transaction_amount',
        'transaction_amount_diff', 'daily_transaction_count'
    ]

# Example usage
if __name__ == "__main__":
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Example transaction
    sample_transaction = {
        'Account No': '1234567890',
        'DATE': '2024-02-11 10:30:00',
        'VALUE DATE': '2024-02-11 10:30:00',
        'WITHDRAWAL AMT': 5000.0,
        'DEPOSIT AMT': 0.0,
        'BALANCE AMT': 10000.0,
        'transaction_amount': 5000.0,
        'TRANSACTION DETAILS': 'Online Transfer',
        'CHQ.NO.': ''
    }
    
    # Make prediction
    features = get_feature_names()
    result = predict_transaction(sample_transaction, model, scaler, features)
    print("\nPrediction Result:")
    print(f"Is Fraudulent: {result['is_fraud']}")
    print(f"Fraud Probability: {result['fraud_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}") 