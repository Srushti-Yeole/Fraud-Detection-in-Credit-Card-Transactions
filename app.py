from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import os
import traceback

# Configure logging with more detail
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,  # Changed to DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Global variables for model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the model and scaler files"""
    global model, scaler
    try:
        if not os.path.exists('models/fraud_detection_model.pkl'):
            logging.info("Model not found. Training new model...")
            from continuous_learning import ContinuousLearningPipeline
            pipeline = ContinuousLearningPipeline()
            pipeline.run_pipeline()
        
        model = joblib.load('models/fraud_detection_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        logging.info("Model and scaler loaded successfully")
        return True
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

# Initialize model and scaler
if not load_model_and_scaler():
    raise RuntimeError("Failed to load model and scaler")

def assess_risk(amount, time_of_day, n_transactions, fraud_probability):
    """Assess risk based on multiple factors"""
    try:
        # Base risk from model probability
        if fraud_probability > 0.5:
            base_risk = 'high'
        elif fraud_probability > 0.2:
            base_risk = 'medium'
        else:
            base_risk = 'low'
        
        # Risk multipliers
        risk_score = fraud_probability
        
        # High amount transactions
        if amount > 10000:
            risk_score *= 1.5
        elif amount > 5000:
            risk_score *= 1.2
        
        # Night time transactions (11 PM - 5 AM)
        if time_of_day < 5 or time_of_day > 23:
            risk_score *= 1.3
        
        # Many recent transactions
        if n_transactions > 15:
            risk_score *= 1.4
        elif n_transactions > 10:
            risk_score *= 1.2
        
        # Final risk assessment
        if risk_score > 0.5:
            return 'high'
        elif risk_score > 0.2:
            return 'medium'
        return 'low'
    except Exception as e:
        logging.error(f"Error in assess_risk: {str(e)}")
        logging.error(f"Inputs: amount={amount}, time={time_of_day}, n_trans={n_transactions}, prob={fraud_probability}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get transaction data
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 400
            }), 400
            
        transaction_data = request.get_json()
        logging.debug(f"Received transaction data: {transaction_data}")
        
        # Validate required fields
        required_fields = ['amount', 'time_of_day', 'n_transactions']
        for field in required_fields:
            if field not in transaction_data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 400
                }), 400
        
        # Extract and validate features
        try:
            amount = float(transaction_data['amount'])
            time_of_day = float(transaction_data['time_of_day'])
            n_transactions = float(transaction_data['n_transactions'])
            
            if amount < 0 or time_of_day < 0 or time_of_day > 24 or n_transactions < 0:
                return jsonify({
                    'error': 'Invalid values: amount and n_transactions must be positive, time_of_day must be between 0 and 24',
                    'status': 400
                }), 400
                
        except ValueError as e:
            logging.error(f"Value error parsing features: {str(e)}")
            return jsonify({
                'error': 'Invalid numeric values provided',
                'status': 400
            }), 400
        
        # Prepare features
        features = pd.DataFrame([{
            'amount': amount,
            'time_of_day': time_of_day,
            'n_transactions': n_transactions
        }])
        
        # Ensure model and scaler are loaded
        global model, scaler
        if model is None or scaler is None:
            if not load_model_and_scaler():
                return jsonify({
                    'error': 'Model not available',
                    'status': 500
                }), 500
        
        # Scale features and make prediction
        try:
            features_scaled = scaler.transform(features)
            probabilities = model.predict_proba(features_scaled)
            fraud_probability = float(probabilities[0][1])
            
            # Assess risk level
            risk_level = assess_risk(amount, time_of_day, n_transactions, fraud_probability)
            
            # Prepare response
            prediction = {
                'fraud_probability': fraud_probability,
                'risk_level': risk_level,
                'timestamp': datetime.now().isoformat()
            }
            
            # Determine if transaction should be blocked
            is_high_risk = risk_level == 'high'
            is_suspicious_amount = amount > 10000
            is_night_time = time_of_day < 5 or time_of_day > 23
            is_many_transactions = n_transactions > 15
            
            prediction['blocked'] = (
                (is_high_risk and is_suspicious_amount) or
                (is_high_risk and is_night_time and is_many_transactions) or
                fraud_probability > 0.7
            )
            
            # Log transaction
            log_entry = {
                'timestamp': prediction['timestamp'],
                'amount': amount,
                'time_of_day': time_of_day,
                'n_transactions': n_transactions,
                'fraud': int(prediction['blocked']),
                'fraud_probability': fraud_probability,
                'risk_level': risk_level
            }
            
            # Save to CSV for persistence
            pd.DataFrame([log_entry]).to_csv(
                'logs/transactions.csv',
                mode='a',
                header=not os.path.exists('logs/transactions.csv'),
                index=False
            )
            
            return jsonify(prediction)
            
        except Exception as e:
            logging.error(f"Model prediction error: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': 'Error making prediction',
                'status': 500
            }), 500
            
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'status': 500
        }), 500

@app.route('/metrics')
def metrics():
    """Endpoint for monitoring metrics"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize empty DataFrame with correct columns if file doesn't exist
        columns = ['timestamp', 'amount', 'time_of_day', 'n_transactions', 
                  'fraud', 'fraud_probability', 'risk_level']
                  
        if not os.path.exists('logs/transactions.csv'):
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv('logs/transactions.csv', index=False)
            logging.info("Created new transactions log file")
            
            # Return empty metrics
            return jsonify({
                'total_predictions': 0,
                'fraud_detected': 0,
                'blocked_transactions': 0,
                'average_fraud_probability': 0.0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0,
                'fraud_rate': 0.0,
                'block_rate': 0.0,
                'recent_transactions': []
            })
        
        # Read existing transactions with error handling for malformed CSV
        try:
            df = pd.read_csv('logs/transactions.csv', on_bad_lines='skip')
            
            # Validate columns
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                logging.error(f"Missing columns in transactions.csv: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    df[col] = ''
            
            # Ensure all required columns exist and have correct types
            df = df[columns]  # Reorder and select only needed columns
            
            # Convert numeric columns
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['time_of_day'] = pd.to_numeric(df['time_of_day'], errors='coerce')
            df['n_transactions'] = pd.to_numeric(df['n_transactions'], errors='coerce')
            df['fraud'] = pd.to_numeric(df['fraud'], errors='coerce')
            df['fraud_probability'] = pd.to_numeric(df['fraud_probability'], errors='coerce')
            
            # Fill NaN values with defaults
            df = df.fillna({
                'amount': 0.0,
                'time_of_day': 0.0,
                'n_transactions': 0,
                'fraud': 0,
                'fraud_probability': 0.0,
                'risk_level': 'low'
            })
            
            # Save cleaned data back to CSV
            df.to_csv('logs/transactions.csv', index=False)
            
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            df.to_csv('logs/transactions.csv', index=False)
        
        if len(df) == 0:
            # Return empty metrics if file exists but is empty
            return jsonify({
                'total_predictions': 0,
                'fraud_detected': 0,
                'blocked_transactions': 0,
                'average_fraud_probability': 0.0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0,
                'fraud_rate': 0.0,
                'block_rate': 0.0,
                'recent_transactions': []
            })
        
        # Convert timestamp strings to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp and get last 10 transactions
        df = df.sort_values('timestamp', ascending=False)
        recent_transactions = df.head(10).to_dict('records')
        
        # Format timestamps for display
        for transaction in recent_transactions:
            transaction['timestamp'] = pd.to_datetime(transaction['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            # Ensure numeric values are properly formatted
            transaction['amount'] = float(transaction['amount'])
            transaction['time_of_day'] = float(transaction['time_of_day'])
            transaction['n_transactions'] = int(float(transaction['n_transactions']))
            transaction['fraud_probability'] = float(transaction['fraud_probability'])
            transaction['fraud'] = int(float(transaction['fraud']))
        
        # Calculate statistics
        stats = {
            'total_predictions': len(df),
            'fraud_detected': int(df['fraud_probability'].gt(0.5).sum()),
            'blocked_transactions': int(df['fraud'].sum()),
            'average_fraud_probability': float(df['fraud_probability'].mean() * 100),
            'high_risk_count': int(df['risk_level'].eq('high').sum()),
            'medium_risk_count': int(df['risk_level'].eq('medium').sum()),
            'low_risk_count': int(df['risk_level'].eq('low').sum()),
            'recent_transactions': recent_transactions
        }
        
        # Add percentage calculations
        if len(df) > 0:
            stats['fraud_rate'] = float(stats['fraud_detected'] / len(df) * 100)
            stats['block_rate'] = float(stats['blocked_transactions'] / len(df) * 100)
        else:
            stats['fraud_rate'] = 0.0
            stats['block_rate'] = 0.0
        
        logging.info("Metrics calculated successfully")
        return jsonify(stats)
        
    except Exception as e:
        logging.error(f"Metrics error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Error fetching metrics: {str(e)}',
            'status': 500
        }), 500

@app.route('/retrain', methods=['POST'])
def trigger_retraining():
    """Endpoint to manually trigger model retraining"""
    try:
        from continuous_learning import ContinuousLearningPipeline
        pipeline = ContinuousLearningPipeline()
        success = pipeline.run_pipeline()
        
        if success:
            # Reload the model and scaler
            global model, scaler
            model = joblib.load('models/fraud_detection_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            return jsonify({'message': 'Retraining completed successfully'})
        else:
            return jsonify({'error': 'Retraining failed'}), 500
            
    except Exception as e:
        logging.error(f"Retraining error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Error during retraining'}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    try:
        # Start in debug mode for better error reporting
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting Flask app: {str(e)}")
        logging.error(f"Flask startup error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise