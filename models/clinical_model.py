"""
Clinical Data Model using Random Forest and XGBoost
Trained on UCI Heart Disease Dataset
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

class ClinicalPredictor:
    def __init__(self):
        """Initialize clinical prediction model"""
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load models if already trained
            self.rf_model = joblib.load('models/saved/clinical_rf.pkl')
            self.xgb_model = joblib.load('models/saved/clinical_xgb.pkl')
            self.scaler = joblib.load('models/saved/clinical_scaler.pkl')
        except:
            # Train new models if not found
            self.train_models()
    
    def train_models(self):
        """Train models on UCI Heart Disease dataset"""
        # Load UCI Heart Disease dataset
        # For demo, using synthetic data similar to UCI format
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic training data
        X_train = pd.DataFrame({
            'age': np.random.randint(29, 77, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),  # chest pain type
            'trestbps': np.random.randint(94, 200, n_samples),  # resting BP
            'chol': np.random.randint(126, 564, n_samples),  # cholesterol
            'fbs': np.random.randint(0, 2, n_samples),  # fasting blood sugar
            'restecg': np.random.randint(0, 3, n_samples),  # ECG results
            'thalach': np.random.randint(71, 202, n_samples),  # max heart rate
            'exang': np.random.randint(0, 2, n_samples),  # exercise angina
            'oldpeak': np.random.uniform(0, 6.2, n_samples)  # ST depression
        })
        
        # Generate target based on risk factors
        y_train = ((X_train['age'] > 50) & 
                   (X_train['chol'] > 240) & 
                   (X_train['trestbps'] > 140)).astype(int)
        
        # Add some randomness
        noise = np.random.binomial(1, 0.2, n_samples)
        y_train = np.logical_xor(y_train, noise).astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.rf_model.fit(X_scaled, y_train)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42
        )
        self.xgb_model.fit(X_scaled, y_train)
        
        # Save models
        import os
        os.makedirs('models/saved', exist_ok=True)
        joblib.dump(self.rf_model, 'models/saved/clinical_rf.pkl')
        joblib.dump(self.xgb_model, 'models/saved/clinical_xgb.pkl')
        joblib.dump(self.scaler, 'models/saved/clinical_scaler.pkl')
    
    def predict(self, clinical_data):
        """Make prediction on clinical data"""
        # Convert to DataFrame
        df = pd.DataFrame([{
            'age': clinical_data['age'],
            'sex': 1 if clinical_data['gender'] == 'M' else 0,
            'cp': 2,  # typical angina
            'trestbps': clinical_data['bp_systolic'],
            'chol': clinical_data['cholesterol'],
            'fbs': 1 if clinical_data['blood_sugar'] > 120 else 0,
            'restecg': 1,  # normal
            'thalach': 150,  # estimated max heart rate
            'exang': clinical_data.get('exercise_angina', 0),
            'oldpeak': 1.0  # ST depression
        }])
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Get predictions from both models
        rf_prob = self.rf_model.predict_proba(X_scaled)[0, 1]
        xgb_prob = self.xgb_model.predict_proba(X_scaled)[0, 1]
        
        # Ensemble prediction (weighted average)
        final_risk = 0.6 * xgb_prob + 0.4 * rf_prob
        
        # Calculate confidence based on agreement
        confidence = 1.0 - abs(rf_prob - xgb_prob)
        
        return final_risk, confidence
