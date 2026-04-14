"""
Multi-Modal Fusion Model
Combines predictions from all models using ensemble learning
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib

class FusionModel:
    def __init__(self):
        """Initialize fusion model"""
        self.weights = {
            'clinical': 0.4,
            'image': 0.35,
            'wearable': 0.25
        }
        self.threshold = 0.5
    
    def predict(self, inputs):
        """
        Combine predictions from multiple models
        Uses confidence-weighted ensemble
        """
        # Extract inputs
        clinical_risk = inputs['clinical_risk']
        clinical_conf = inputs['clinical_confidence']
        image_risk = inputs['image_risk']
        image_conf = inputs['image_confidence']
        wearable_risk = inputs['wearable_risk']
        wearable_conf = inputs['wearable_confidence']
        
        # Method 1: Weighted average based on confidence
        total_conf = clinical_conf + image_conf + wearable_conf
        
        if total_conf > 0:
            weighted_risk = (
                (clinical_risk * clinical_conf * self.weights['clinical']) +
                (image_risk * image_conf * self.weights['image']) +
                (wearable_risk * wearable_conf * self.weights['wearable'])
            ) / (
                (clinical_conf * self.weights['clinical']) +
                (image_conf * self.weights['image']) +
                (wearable_conf * self.weights['wearable'])
            )
        else:
            # Fallback to simple weighted average
            weighted_risk = (
                clinical_risk * self.weights['clinical'] +
                image_risk * self.weights['image'] +
                wearable_risk * self.weights['wearable']
            )
        
        # Method 2: Check agreement between models
        risks = [clinical_risk, image_risk, wearable_risk]
        risks = [r for r in risks if r > 0]  # Filter out zero risks
        
        if len(risks) > 1:
            agreement = 1 - np.std(risks)
        else:
            agreement = 0.5
        
        # Final confidence is based on individual confidences and agreement
        final_confidence = np.mean([clinical_conf, image_conf, wearable_conf]) * agreement
        
        # Apply non-linear transformation for better risk stratification
        if weighted_risk > 0.7:
            weighted_risk = weighted_risk * 1.1  # Amplify high risks
        elif weighted_risk < 0.3:
            weighted_risk = weighted_risk * 0.9  # Reduce low risks
        
        # Ensure risk is between 0 and 1
        final_risk = np.clip(weighted_risk, 0, 1)
        
        return float(final_risk), float(final_confidence)
    
    def get_model_contributions(self, inputs):
        """Get contribution of each model to final prediction"""
        contributions = {
            'clinical': inputs['clinical_risk'] * self.weights['clinical'],
            'image': inputs['image_risk'] * self.weights['image'],
            'wearable': inputs['wearable_risk'] * self.weights['wearable']
        }
        return contributions
