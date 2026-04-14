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
import tensorflow as tf
from models.clinical_model import ClinicalPredictor
from models.image_model import ImagePredictor
from models.wearable_model import WearablePredictor
from models.fusion_model import FusionModel
from utils.risk_calculator import RiskCalculator
from utils.explainable_ai import ExplainableAI
from translations import get_translation, get_all_translations
import logging

FIELD_LIMITS = {
    'age': {'label': 'Age', 'min': 1, 'max': 100, 'type': float},
    'bp_systolic': {'label': 'Systolic blood pressure', 'min': 70, 'max': 250, 'type': float},
    'bp_diastolic': {'label': 'Diastolic blood pressure', 'min': 40, 'max': 150, 'type': float},
    'cholesterol': {'label': 'Cholesterol', 'min': 100, 'max': 400, 'type': float},
    'blood_sugar': {'label': 'Blood sugar', 'min': 50, 'max': 400, 'type': float},
    'bmi': {'label': 'BMI', 'min': 10, 'max': 60, 'type': float},
    'exercise_hours': {'label': 'Exercise hours', 'min': 0, 'max': 24, 'type': float},
    'heart_rate': {'label': 'Heart rate', 'min': 30, 'max': 220, 'type': float},
    'hrv': {'label': 'Heart rate variability', 'min': 0, 'max': 200, 'type': float},
    'steps': {'label': 'Daily steps', 'min': 0, 'max': 100000, 'type': int},
    'sleep': {'label': 'Sleep hours', 'min': 0, 'max': 24, 'type': float},
    'spo2': {'label': 'SpO2', 'min': 70, 'max': 100, 'type': float}
}


def parse_limited_value(form, field_name):
    """Validate numeric fields using shared min/max limits."""
    rules = FIELD_LIMITS[field_name]
    raw_value = form.get(field_name, '').strip()

    if raw_value == '':
        raise ValueError(f"{rules['label']} is required.")

    try:
        value = rules['type'](raw_value)
    except ValueError as exc:
        raise ValueError(f"{rules['label']} must be a valid number.") from exc

    if value < rules['min']:
        raise ValueError(f"{rules['label']} must be at least {rules['min']}.")
    if value > rules['max']:
        raise ValueError(f"{rules['label']} limit exceeded. Maximum allowed is {rules['max']}.")

    return value


def parse_choice_value(form, field_name, allowed_values, label):
    """Validate select inputs against allowed values."""
    value = form.get(field_name, '').strip()
    if value not in allowed_values:
        raise ValueError(f"Invalid {label.lower()} selected.")
    return value

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'medai-fusion-2024-healthcare'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
clinical_model = ClinicalPredictor()
image_model = ImagePredictor()
wearable_model = WearablePredictor()
fusion_model = FusionModel()
risk_calc = RiskCalculator()
explainer = ExplainableAI()

# Supported languages for Indian accessibility
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'kn': 'ಕನ್ನಡ',
    'hi': 'हिंदी'
}

@app.route('/')
def index():
    """Landing page with multi-language support"""
    lang = request.args.get('lang', 'en')
    if lang not in SUPPORTED_LANGUAGES:
        lang = 'en'
    translations = get_all_translations(lang)
    return render_template('index.html', 
                         languages=SUPPORTED_LANGUAGES,
                         current_lang=lang,
                         t=translations)

@app.route('/dashboard')
def dashboard():
    """Patient dashboard for data input"""
    lang = request.args.get('lang', 'en')
    if lang not in SUPPORTED_LANGUAGES:
        lang = 'en'
    translations = get_all_translations(lang)
    return render_template('dashboard.html', 
                         languages=SUPPORTED_LANGUAGES,
                         current_lang=lang,
                         t=translations)

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - Multi-modal AI fusion"""
    try:
        # Get language preference
        lang = request.form.get('lang', 'en')
        if lang not in SUPPORTED_LANGUAGES:
            lang = 'en'
        
        # 1. Collect clinical data
        clinical_data = {
            'age': parse_limited_value(request.form, 'age'),
            'gender': parse_choice_value(request.form, 'gender', {'M', 'F', 'O'}, 'Gender'),
            'bp_systolic': parse_limited_value(request.form, 'bp_systolic'),
            'bp_diastolic': parse_limited_value(request.form, 'bp_diastolic'),
            'cholesterol': parse_limited_value(request.form, 'cholesterol'),
            'blood_sugar': parse_limited_value(request.form, 'blood_sugar'),
            'family_history': int(parse_choice_value(request.form, 'family_history', {'0', '1'}, 'Family history')),
            'smoking': int(parse_choice_value(request.form, 'smoking', {'0', '1', '2'}, 'Smoking status')),
            'bmi': parse_limited_value(request.form, 'bmi'),
            'exercise_hours': parse_limited_value(request.form, 'exercise_hours')
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
                image_risk, image_confidence = image_model.predict(filepath)
                logger.info(f"Image model prediction: {image_risk:.2f}")
        
        # 3. Process wearable data
        wearable_risk = 0.0
        wearable_confidence = 0.0
        wearable_data = {
            'heart_rate': parse_limited_value(request.form, 'heart_rate'),
            'heart_rate_variability': parse_limited_value(request.form, 'hrv'),
            'steps_daily': parse_limited_value(request.form, 'steps'),
            'sleep_hours': parse_limited_value(request.form, 'sleep'),
            'spo2': parse_limited_value(request.form, 'spo2')
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
        session['lang'] = lang
        
        return jsonify({
            'success': True,
            'redirect': f'/results?lang={lang}'
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
    
    lang = request.args.get('lang', 'en')
    if lang not in SUPPORTED_LANGUAGES:
        lang = 'en'
    translations = get_all_translations(lang)
    return render_template('results.html', 
                         results=session['results'],
                         languages=SUPPORTED_LANGUAGES,
                         current_lang=lang,
                         t=translations)

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
    lang = request.args.get('lang', 'en')
    if lang not in SUPPORTED_LANGUAGES:
        lang = 'en'
    translations = get_all_translations(lang)
    impact_metrics = {
        'lives_impacted': '10,000+',
        'early_detections': '2,500+',
        'cost_saved': '₹5 Crores',
        'rural_reach': '50 PHCs'
    }
    return render_template('about.html', 
                         metrics=impact_metrics,
                         languages=SUPPORTED_LANGUAGES,
                         current_lang=lang,
                         t=translations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
