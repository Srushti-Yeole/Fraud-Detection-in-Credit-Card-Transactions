# Credit Card Fraud Detection System

A robust, real-time credit card fraud detection system built with Python, utilizing machine learning and continuous learning capabilities.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-orange.svg)
![Monitoring](https://img.shields.io/badge/Monitoring-Prometheus-red.svg)

## 🌟 Features

- **Real-time Fraud Detection**
  - Instant transaction analysis
  - Multi-factor risk assessment
  - Configurable blocking rules
  - JSON API interface

- **Advanced Machine Learning**
  - Random Forest Classification
  - Balanced class handling
  - Feature scaling and normalization
  - Synthetic data generation for training

- **Continuous Learning Pipeline**
  - Automated model retraining
  - Performance validation
  - Model versioning
  - Data persistence

- **Production-Ready Monitoring**
  - Prometheus metrics integration
  - Transaction monitoring
  - Performance tracking
  - System health checks

## 🚀 Quick Start

1. **Clone the repository**
```bash
git https://github.com/Ronak-599/Fraud-Detection-in-Credit-Card-Transactions
cd Credit-Card-Fraud-Detection
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## 📊 System Architecture

The system consists of four main components:

1. **Data Processing Pipeline**
   - Feature engineering
   - Data validation
   - Preprocessing
   - Scaling

2. **Model Development**
   - Random Forest Classifier
   - Cross-validation
   - Performance metrics
   - Model persistence

3. **Real-time Detection System**
   - REST API endpoints
   - Risk scoring
   - Transaction validation
   - Blocking rules

4. **Monitoring System**
   - Prometheus metrics
   - Performance tracking
   - System logging
   - Audit trail

## 🔧 API Endpoints

### Predict Endpoint
```bash
POST /predict
Content-Type: application/json

{
    "amount": 1000.0,
    "time_of_day": 14.5,
    "n_transactions": 5
}
```

### Response Format
```json
{
    "fraud_probability": 0.15,
    "risk_level": "low",
    "blocked": false,
    "timestamp": "2024-03-21T14:30:00Z"
}
```

## 📈 Monitoring Metrics

- Total predictions made
- Fraud detections
- Blocked transactions
- Response times
- Model latency
- System health

## 🛠️ Configuration

The system can be configured through environment variables:

- `MODEL_THRESHOLD`: Fraud probability threshold
- `MONITORING_PORT`: Prometheus metrics port
- `MIN_SAMPLES`: Minimum samples for retraining
- `PERFORMANCE_THRESHOLD`: Model performance threshold

## 🔍 Model Performance

- Accuracy: 95%+ on test set
- Balanced handling of fraud cases
- Real-time prediction capability
- Continuous performance monitoring

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request




## 🙏 Acknowledgments

- Scikit-learn team for machine learning tools
- Flask team for the web framework
- Prometheus team for monitoring capabilities 