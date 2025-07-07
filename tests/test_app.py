import pytest
import json
import numpy as np
from app import app, redis_client, monitor
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_redis():
    with patch('app.redis_client') as mock:
        yield mock

@pytest.fixture
def mock_monitor():
    with patch('app.monitor') as mock:
        yield mock

@pytest.fixture
def sample_transaction():
    return {
        'transaction_id': 'test123',
        'amount': 1000.00,
        'merchant_id': 'merchant123',
        'customer_id': 'customer456',
        'v1': 1.0,
        'v2': -0.5,
        'v3': 1.2
    }

def test_home_page(client):
    """Test home page loads correctly"""
    response = client.get('/')
    assert response.status_code == 200

def test_predict_endpoint_success(client, sample_transaction, mock_redis, mock_monitor):
    """Test successful prediction request"""
    # Mock Redis cache miss
    mock_redis.get.return_value = None
    
    # Mock model prediction
    with patch('app.model.predict_proba') as mock_predict:
        mock_predict.return_value = np.array([[0.1, 0.9]])  # High fraud probability
        
        response = client.post(
            '/predict',
            data=json.dumps(sample_transaction),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'fraud_probability' in data
        assert 'risk_level' in data
        assert data['blocked'] == True  # Should be blocked due to high probability
        
        # Verify monitoring calls
        mock_monitor.record_prediction.assert_called_once()
        mock_monitor.record_blocked_transaction.assert_called_once()

def test_predict_endpoint_cached(client, sample_transaction, mock_redis):
    """Test prediction with cached result"""
    cached_result = {
        'fraud_probability': 0.8,
        'risk_level': 'high',
        'blocked': True,
        'timestamp': '2023-01-01T00:00:00'
    }
    mock_redis.get.return_value = json.dumps(cached_result)
    
    response = client.post(
        '/predict',
        data=json.dumps(sample_transaction),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data == cached_result

def test_rate_limiting(client, sample_transaction, mock_redis):
    """Test rate limiting functionality"""
    # Mock rate limit exceeded
    mock_redis.pipeline.return_value.execute.return_value = [None, None, 101]  # Over limit
    
    response = client.post(
        '/predict',
        data=json.dumps(sample_transaction),
        content_type='application/json'
    )
    
    assert response.status_code == 429
    data = json.loads(response.data)
    assert data['error'] == 'Rate limit exceeded'

def test_metrics_endpoint(client, mock_redis):
    """Test metrics endpoint"""
    # Mock Redis recent transactions
    mock_transactions = [
        json.dumps({
            'timestamp': '2023-01-01T00:00:00',
            'transaction_id': 'tx1',
            'amount': 1000,
            'fraud_probability': 0.9,
            'blocked': True
        })
    ]
    mock_redis.lrange.return_value = mock_transactions
    
    # Mock monitor metrics
    monitor.PREDICTIONS_TOTAL._value.get.return_value = 100
    monitor.FRAUD_DETECTED._value.get.return_value = 10
    monitor.BLOCKED_TRANSACTIONS._value.get.return_value = 5
    monitor.RESPONSE_TIME._sum.get.return_value = 10
    monitor.RESPONSE_TIME._count.get.return_value = 100
    
    response = client.get('/metrics')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'total_predictions' in data
    assert 'fraud_detected' in data
    assert 'blocked_transactions' in data
    assert 'average_response_time' in data
    assert 'recent_transactions' in data

def test_retrain_endpoint(client):
    """Test model retraining endpoint"""
    with patch('app.pipeline.run_pipeline') as mock_pipeline:
        response = client.post('/retrain')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Retraining started'
        mock_pipeline.assert_called_once()

def test_invalid_transaction_data(client):
    """Test handling of invalid transaction data"""
    response = client.post(
        '/predict',
        data='invalid json',
        content_type='application/json'
    )
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data

def test_missing_required_fields(client):
    """Test handling of missing required fields"""
    invalid_transaction = {'amount': 1000}  # Missing other required fields
    
    response = client.post(
        '/predict',
        data=json.dumps(invalid_transaction),
        content_type='application/json'
    )
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data

def test_high_amount_medium_risk(client, mock_redis, mock_monitor):
    """Test blocking of high amount transactions with medium risk"""
    transaction = {
        'transaction_id': 'test123',
        'amount': 15000.00,  # High amount
        'merchant_id': 'merchant123',
        'customer_id': 'customer456'
    }
    
    # Mock Redis cache miss
    mock_redis.get.return_value = None
    
    # Mock model prediction with medium risk
    with patch('app.model.predict_proba') as mock_predict:
        mock_predict.return_value = np.array([[0.3, 0.7]])  # Medium fraud probability
        
        response = client.post(
            '/predict',
            data=json.dumps(transaction),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['blocked'] == True  # Should be blocked due to high amount
        assert data['risk_level'] == 'medium'

def test_metrics_error_handling(client, mock_redis):
    """Test error handling in metrics endpoint"""
    # Mock Redis error
    mock_redis.lrange.side_effect = Exception('Redis error')
    
    response = client.get('/metrics')
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data

def test_retrain_error_handling(client):
    """Test error handling in retrain endpoint"""
    with patch('app.pipeline.run_pipeline', side_effect=Exception('Training error')):
        response = client.post('/retrain')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data 