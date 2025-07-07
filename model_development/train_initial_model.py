import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create a simple initial model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train on some dummy data
X = np.random.rand(1000, 30)  # 30 features
y = np.random.randint(0, 2, 1000)  # Binary classification
model.fit(X, y)

# Ensure models directory exists
os.makedirs('../models', exist_ok=True)

# Save the model
joblib.dump(model, '../models/fraud_detection_model.pkl') 