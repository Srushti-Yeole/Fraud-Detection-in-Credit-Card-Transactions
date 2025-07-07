import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create synthetic data for initial model
def create_synthetic_data():
    np.random.seed(42)
    n_samples = 10000
    
    # Generate features
    amount = np.random.lognormal(mean=4, sigma=1, size=n_samples)
    time = np.random.uniform(0, 24, n_samples)
    n_transactions = np.random.poisson(lam=5, size=n_samples)
    
    # Create fraud cases (5% of transactions)
    fraud = np.zeros(n_samples)
    fraud_idx = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    fraud[fraud_idx] = 1
    
    # Create DataFrame
    data = pd.DataFrame({
        'amount': amount,
        'time_of_day': time,
        'n_transactions': n_transactions,
        'fraud': fraud
    })
    
    return data

def train_model():
    # Create data
    data = create_synthetic_data()
    
    # Split features and target
    X = data.drop('fraud', axis=1)
    y = data['fraud']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fraud_detection_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

if __name__ == '__main__':
    train_model() 