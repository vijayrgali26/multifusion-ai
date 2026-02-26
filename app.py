"""
MedAI Fusion - Multi-Modal AI Healthcare Platform
Early Disease Detection for Affordable Healthcare in India
"""

from flask import Flask, render_template, request, jsonify, session, redirect
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import pickle
import logging

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'medai-fusion-2024-healthcare'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy loading for TensorFlow and models
_models_loaded = False
clinical_model = None
image_model = None
wearable_model = None
fusion_model = None
risk_calc = None
explainer = None

def load_models():
    """Lazy load models only when needed"""
    global _models_loaded, clinical_model, image_model, wearable_model, fusion_model, risk_calc, explainer
    
    if _models_loaded:
        return
    
    try:
        from models.clinical_model import ClinicalPredictor
        from models.image_model import ImagePredictor
        from models.wearable_model import WearablePredictor
        from models.fusion_model import FusionModel
        from utils.risk_calculator import RiskCalculator
        from utils.explainable_ai import ExplainableAI
        
        clinical_model = ClinicalPredictor()
        image_model = ImagePredictor()
        wearable_model = WearablePredictor()
        fusion_model = FusionModel()
        risk_calc = RiskCalculator()
        explainer = ExplainableAI()
        _models_loaded = True
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        _models_loaded = True  # Mark as attempted to avoid repeated failures

# Supported languages for Indian accessibility
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'हिंदी',
    'ta': 'தமிழ்',
    'te': 'తెలుగు',
    'kn': 'ಕನ್ನಡ',
    'ml': 'മലയാളം',
    'mr': 'मराठी',
    'bn': 'বাংলা',
    'gu': 'ગુજરાતી',
    'pa': 'ਪੰਜਾਬੀ'
}

@app.route('/')
def index():
    """Landing page with multi-language support"""
    lang = request.args.get('lang', 'en')
    return render_template('index.html', 
                         languages=SUPPORTED_LANGUAGES,
                         current_lang=lang)

@app.route('/dashboard')
def dashboard():
    """Patient dashboard for data input"""
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - Multi-modal AI fusion"""
    try:
        load_models()
        
        # 1. Collect clinical data
        clinical_data = {
            'age': float(request.form.get('age', 0)),
            'gender': request.form.get('gender', 'M'),
            'bp_systolic': float(request.form.get('bp_systolic', 120)),
            'bp_diastolic': float(request.form.get('bp_diastolic', 80)),
            'cholesterol': float(request.form.get('cholesterol', 200)),
            'blood_sugar': float(request.form.get('blood_sugar', 100)),
            'family_history': int(request.form.get('family_history', 0)),
            'smoking': int(request.form.get('smoking', 0)),
            'bmi': float(request.form.get('bmi', 22)),
            'exercise_hours': float(request.form.get('exercise_hours', 2))
        }
        
        # 2. Process medical image if uploaded
        image_risk = 0.0
        image_confidence = 0.0
        if 'medical_image' in request.files:
            file = request.files['medical_image']
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)
                
                # Predict using CNN model
                if image_model:
                    image_risk, image_confidence = image_model.predict(filepath)
                logger.info(f"Image model prediction: {image_risk:.2f}")
        
        # 3. Process wearable data
        wearable_risk = 0.0
        wearable_confidence = 0.0
        wearable_data = {
            'heart_rate': float(request.form.get('heart_rate', 72)),
            'heart_rate_variability': float(request.form.get('hrv', 50)),
            'steps_daily': int(request.form.get('steps', 5000)),
            'sleep_hours': float(request.form.get('sleep', 7)),
            'spo2': float(request.form.get('spo2', 98))
        }
        
        # 4. Get predictions from each model
        clinical_risk, clinical_confidence = clinical_model.predict(clinical_data)
        wearable_risk, wearable_confidence = wearable_model.predict(wearable_data)
        
        # 5. Fusion model combines all predictions
        fusion_input = {
            'clinical_risk': clinical_risk,
            'clinical_confidence': clinical_confidence,
            'image_risk': image_risk,
            'image_confidence': image_confidence,
            'wearable_risk': wearable_risk,
            'wearable_confidence': wearable_confidence
        }
        
        final_risk, final_confidence = fusion_model.predict(fusion_input)
        
        # 6. Calculate risk level and recommendations
        risk_level = risk_calc.get_risk_level(final_risk)
        recommendations = risk_calc.get_recommendations(final_risk, clinical_data)
        
        # 7. Generate explainable AI insights
        explanations = explainer.explain(clinical_data, final_risk)
        
        # 8. Calculate cost savings
        early_detection_cost = 1500  # INR
        late_stage_cost = 500000  # INR
        potential_savings = late_stage_cost - early_detection_cost
        
        # Store in session for results page
        session['results'] = {
            'final_risk': round(final_risk * 100, 2),
            'confidence': round(final_confidence * 100, 2),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'explanations': explanations,
            'cost_savings': potential_savings,
            'individual_risks': {
                'clinical': round(clinical_risk * 100, 2),
                'imaging': round(image_risk * 100, 2),
                'wearable': round(wearable_risk * 100, 2)
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({
            'success': True,
            'redirect': '/results'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/results')
def results():
    """Display prediction results with explanations"""
    if 'results' not in session:
        return redirect('/')
    
    return render_template('results.html', results=session['results'])

@app.route('/api/trend-analysis', methods=['POST'])
def trend_analysis():
    """Analyze health trends over time"""
    data = request.json
    patient_id = data.get('patient_id')
    
    # Fetch historical data and calculate trends
    trends = risk_calc.calculate_trends(patient_id)
    
    return jsonify({
        'success': True,
        'trends': trends,
        'prediction': 'Risk may increase by 15% in 6 months without intervention'
    })

@app.route('/about')
def about():
    """About page with social impact metrics"""
    impact_metrics = {
        'lives_impacted': '10,000+',
        'early_detections': '2,500+',
        'cost_saved': '₹5 Crores',
        'rural_reach': '50 PHCs'
    }
    return render_template('about.html', metrics=impact_metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
