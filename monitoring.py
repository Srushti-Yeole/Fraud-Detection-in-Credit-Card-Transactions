from prometheus_client import Counter, Histogram, start_http_server, REGISTRY, CollectorRegistry
import threading
import time

class FraudDetectionMonitor:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FraudDetectionMonitor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # Create a new registry for our metrics
            self.registry = CollectorRegistry()
            
            # Initialize Prometheus metrics with our custom registry
            self.PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions made', registry=self.registry)
            self.FRAUD_DETECTED = Counter('fraud_detected', 'Number of fraudulent transactions detected', registry=self.registry)
            self.BLOCKED_TRANSACTIONS = Counter('blocked_transactions', 'Number of transactions blocked', registry=self.registry)
            self.RESPONSE_TIME = Histogram('response_time_seconds', 'Response time for predictions', registry=self.registry)
            self.MODEL_LATENCY = Histogram('model_latency_seconds', 'Model prediction latency', registry=self.registry)
            
            self._initialized = True
    
    def start(self, port=8000):
        """Start the monitoring server"""
        try:
            # Start server with our custom registry
            start_http_server(port, registry=self.registry)
        except Exception as e:
            print(f"Error starting monitoring server: {str(e)}")
    
    def record_prediction(self, response_time, is_fraud, fraud_probability, amount):
        """Record prediction metrics"""
        try:
            self.PREDICTIONS_TOTAL.inc()
            self.RESPONSE_TIME.observe(response_time)
            if is_fraud:
                self.FRAUD_DETECTED.inc()
        except Exception as e:
            print(f"Error recording prediction metrics: {str(e)}")
    
    def record_blocked_transaction(self, amount):
        """Record blocked transaction"""
        try:
            self.BLOCKED_TRANSACTIONS.inc()
        except Exception as e:
            print(f"Error recording blocked transaction: {str(e)}")

# Create a global monitor instance
monitor = FraudDetectionMonitor() 