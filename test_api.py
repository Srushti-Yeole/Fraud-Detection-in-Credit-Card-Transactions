import requests
import json
import time

def test_prediction():
    # Test data
    transaction = {
        'amount': 1000.0,
        'time_of_day': 14.5,
        'n_transactions': 3
    }
    
    # Make prediction request
    response = requests.post('http://localhost:5000/predict', json=transaction)
    print("\nPrediction Response:")
    print(json.dumps(response.json(), indent=2))
    
    # Get metrics
    time.sleep(1)  # Wait for metrics to update
    response = requests.get('http://localhost:5000/metrics')
    print("\nMetrics Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == '__main__':
    test_prediction() 