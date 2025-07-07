import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
import os

class ContinuousLearningPipeline:
    def __init__(self):
        self.model_path = 'models/fraud_detection_model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.data_path = 'logs/transactions.csv'
        self.min_samples = 1000
        self.performance_threshold = 0.95
        
        # Ensure directories exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename='continuous_learning.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def create_synthetic_data(self, n_samples=10000):
        """Create synthetic data for training"""
        np.random.seed(42)
        
        # Generate normal transactions (95%)
        normal_samples = int(n_samples * 0.95)
        normal_data = pd.DataFrame({
            'amount': np.random.lognormal(mean=4, sigma=1, size=normal_samples),  # Most transactions < 1000
            'time_of_day': np.random.normal(loc=14, scale=3, size=normal_samples).clip(0, 24),  # Business hours
            'n_transactions': np.random.poisson(lam=2, size=normal_samples),  # Few transactions
            'fraud': 0
        })
        
        # Generate fraudulent transactions (5%)
        fraud_samples = n_samples - normal_samples
        
        # Create different types of fraud patterns
        fraud_data_list = []
        
        # Pattern 1: High amount, night time, many transactions
        n1 = fraud_samples // 3
        fraud_data_list.append(pd.DataFrame({
            'amount': np.random.lognormal(mean=9, sigma=1, size=n1),  # High amounts (> 5000)
            'time_of_day': np.random.normal(loc=2, scale=1, size=n1).clip(0, 24),  # Night time
            'n_transactions': np.random.poisson(lam=15, size=n1) + 10,  # Many transactions
            'fraud': 1
        }))
        
        # Pattern 2: Very high amount, any time
        n2 = fraud_samples // 3
        fraud_data_list.append(pd.DataFrame({
            'amount': np.random.lognormal(mean=10, sigma=1, size=n2),  # Very high amounts (> 10000)
            'time_of_day': np.random.uniform(0, 24, size=n2),
            'n_transactions': np.random.poisson(lam=5, size=n2),
            'fraud': 1
        }))
        
        # Pattern 3: Moderate amount but suspicious timing and frequency
        n3 = fraud_samples - n1 - n2
        fraud_data_list.append(pd.DataFrame({
            'amount': np.random.lognormal(mean=6, sigma=1, size=n3),
            'time_of_day': np.random.choice([np.random.normal(2, 1), np.random.normal(23, 1)], size=n3).clip(0, 24),
            'n_transactions': np.random.poisson(lam=10, size=n3) + 5,
            'fraud': 1
        }))
        
        # Combine all fraud patterns
        fraud_data = pd.concat(fraud_data_list, ignore_index=True)
        
        # Combine normal and fraud data
        data = pd.concat([normal_data, fraud_data], ignore_index=True)
        
        # Add some noise to make patterns less perfect
        data['amount'] *= np.random.normal(1, 0.1, size=len(data))
        data['time_of_day'] += np.random.normal(0, 0.5, size=len(data))
        data['time_of_day'] = data['time_of_day'].clip(0, 24)
        
        # Log some statistics
        logging.info(f"Created synthetic dataset with {len(data)} samples")
        logging.info(f"Fraud ratio: {data['fraud'].mean():.2%}")
        
        return data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def validate_model(self, model, X_test, y_test):
        """Validate model performance on test set"""
        try:
            # Basic accuracy
            accuracy = model.score(X_test, y_test)
            logging.info(f"Model accuracy: {accuracy:.2%}")
            
            # Predict probabilities
            y_pred_proba = model.predict_proba(X_test)
            
            # Test some specific cases
            test_cases = [
                {'amount': 15000, 'time_of_day': 2, 'n_transactions': 20},  # High-risk case
                {'amount': 100, 'time_of_day': 14, 'n_transactions': 2}     # Low-risk case
            ]
            
            for case in test_cases:
                X_case = pd.DataFrame([case])
                X_case_scaled = self.scaler.transform(X_case)
                prob = model.predict_proba(X_case_scaled)[0][1]
                logging.info(f"Test case {case}: Fraud probability = {prob:.2%}")
            
            return accuracy >= self.performance_threshold
            
        except Exception as e:
            logging.error(f"Error in model validation: {str(e)}")
            return False
    
    def train_model(self, X, y):
        """Train a new model"""
        try:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X, y)
            return model
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return None
    
    def run_pipeline(self):
        """Run the continuous learning pipeline"""
        try:
            # Create synthetic data
            df = self.create_synthetic_data()
            
            # Prepare features and target
            X = df[['amount', 'time_of_day', 'n_transactions']]
            y = df['fraud']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = self.train_model(X_train_scaled, y_train)
            if model is None:
                return False
            
            # Validate model
            if not self.validate_model(model, X_test_scaled, y_test):
                logging.error("Model failed validation")
                return False
            
            # Save model and scaler
            joblib.dump(model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            logging.info("Model and scaler saved successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error in continuous learning pipeline: {str(e)}")
            return False

if __name__ == '__main__':
    pipeline = ContinuousLearningPipeline()
    pipeline.run_pipeline()