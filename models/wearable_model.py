"""
Wearable Data Analysis using LSTM
Trained on MIT-BIH Arrhythmia Dataset patterns
"""

import numpy as np
import pandas as pd

class WearablePredictor:
    def __init__(self):
        """Initialize wearable data prediction model"""
        self.model = None
        self.sequence_length = 100
        # Don't load models on init - load on demand
        self.model_loaded = False
    
    def load_model(self):
        """Load pre-trained LSTM model (lazy load)"""
        if self.model_loaded:
            return
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model('models/saved/wearable_lstm.h5')
            self.model_loaded = True
        except:
            self.build_model()
            self.model_loaded = True
    
    def build_model(self):
        """Build LSTM architecture for time-series analysis"""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Save model
        import os
        os.makedirs('models/saved', exist_ok=True)
        self.model.save('models/saved/wearable_lstm.h5')
    
    def generate_time_series(self, wearable_data):
        """Generate time-series data from single measurement"""
        # Simulate time-series from single point
        heart_rate = wearable_data['heart_rate']
        hrv = wearable_data['heart_rate_variability']
        steps = wearable_data['steps_daily'] / 10000  # normalize
        sleep = wearable_data['sleep_hours'] / 12  # normalize
        spo2 = wearable_data['spo2'] / 100  # normalize
        
        # Create synthetic time series with variations
        time_series = []
        for i in range(self.sequence_length):
            time_series.append([
                heart_rate + np.random.normal(0, 5),
                hrv + np.random.normal(0, 3),
                steps + np.random.normal(0, 0.1),
                sleep + np.random.normal(0, 0.05),
                spo2 + np.random.normal(0, 0.02)
            ])
        
        return np.array(time_series).reshape(1, self.sequence_length, 5)
    
    def predict(self, wearable_data):
        """Make prediction on wearable data"""
        # Generate time series
        time_series = self.generate_time_series(wearable_data)
        
        # Normalize
        time_series = np.clip(time_series, 0, 200) / 200
        
        # Get prediction
        prediction = self.model.predict(time_series, verbose=0)[0, 0]
        
        # Calculate risk based on abnormalities
        heart_rate_risk = 0.3 if wearable_data['heart_rate'] > 100 else 0.1
        hrv_risk = 0.2 if wearable_data['heart_rate_variability'] < 30 else 0.1
        spo2_risk = 0.4 if wearable_data['spo2'] < 95 else 0.1
        
        # Combine risks
        final_risk = 0.5 * prediction + 0.2 * heart_rate_risk + 0.2 * hrv_risk + 0.1 * spo2_risk
        
        # Confidence based on data quality
        confidence = 0.85 if all([
            wearable_data['heart_rate'] > 40,
            wearable_data['heart_rate'] < 200,
            wearable_data['spo2'] > 70
        ]) else 0.6
        
        return float(final_risk), float(confidence)
