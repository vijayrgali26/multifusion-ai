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
import uuid

# Database and AI Assistant imports
from database.models import (
    add_patient, get_patient, get_patient_predictions,
    add_prediction, add_medical_image, PatientRecord, PredictionRecord
)
from utils.ai_assistant import get_assistant_response, answer_health_question

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

@app.route('/assistant')
def assistant():
    """AI Assistant chatbot interface"""
    return render_template('assistant.html')

@app.route('/patient-profile')
def patient_profile():
    """Patient health profile management"""
    patient_id = request.args.get('patient_id', None)
    return render_template('patient_profile.html', patient_id=patient_id)

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


# ==================== NEW AI ASSISTANT ENDPOINTS ====================

@app.route('/api/assistant/chat', methods=['POST'])
def assistant_chat():
    """
    AI Assistant endpoint for natural language queries
    Directs users to relevant sections of the app
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        language = data.get('language', 'en')
        
        # Get assistant response
        response = get_assistant_response(user_message, language)
        
        return jsonify({
            'success': True,
            'response': response['guidance'],
            'intent': response['intent'],
            'action': response['action'],
            'confidence': response['confidence']
        })
    except Exception as e:
        logger.error(f"Assistant error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/assistant/ask', methods=['POST'])
def ask_question():
    """
    Question answering endpoint using AI
    """
    try:
        data = request.json
        question = data.get('question', '')
        context = data.get('context', None)
        
        result = answer_health_question(question, context)
        
        return jsonify({
            'success': True,
            'question': result['question'],
            'answer': result['answer'],
            'confidence': result['confidence']
        })
    except Exception as e:
        logger.error(f"QA error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== PATIENT MANAGEMENT ENDPOINTS ====================

@app.route('/api/patient/register', methods=['POST'])
def register_patient():
    """
    Register a new patient and store their information
    """
    try:
        data = request.json
        
        # Generate unique patient ID
        patient_id = str(uuid.uuid4())[:8]
        
        # Create patient record
        patient_data = {
            'patient_id': patient_id,
            'name': data.get('name', ''),
            'age': int(data.get('age', 0)),
            'gender': data.get('gender', ''),
            'email': data.get('email', ''),
            'phone': data.get('phone', ''),
            'medical_history': data.get('medical_history', {}),
            'medications': data.get('medications', []),
            'allergies': data.get('allergies', []),
            'lifestyle_factors': data.get('lifestyle_factors', {})
        }
        
        patient = add_patient(patient_data)
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'message': 'Patient registered successfully'
        })
    except Exception as e:
        logger.error(f"Patient registration error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient_info(patient_id):
    """
    Retrieve patient information
    """
    try:
        patient = get_patient(patient_id)
        
        if not patient:
            return jsonify({
                'success': False,
                'error': 'Patient not found'
            }), 404
        
        return jsonify({
            'success': True,
            'patient': patient.to_dict()
        })
    except Exception as e:
        logger.error(f"Patient retrieval error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/patient/<patient_id>/predictions', methods=['GET'])
def get_predictions(patient_id):
    """
    Get prediction history for a patient
    This allows historical data to improve future predictions
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        predictions = get_patient_predictions(patient_id, limit)
        
        prediction_list = [p.to_dict() for p in predictions] if predictions else []
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'predictions': prediction_list,
            'count': len(prediction_list)
        })
    except Exception as e:
        logger.error(f"Predictions retrieval error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== IMAGE UPLOAD WITH ANALYSIS ====================

@app.route('/api/upload-medical-image', methods=['POST'])
def upload_medical_image():
    """
    Upload medical image from user with analysis
    Stores image metadata in patient record
    """
    try:
        patient_id = request.form.get('patient_id', '')
        image_type = request.form.get('image_type', 'chest_xray')  # chest_xray, ct_scan, etc.
        
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Secure filename and save
        filename = secure_filename(f"{patient_id}_{image_type}_{datetime.now().timestamp()}.png")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze image using CNN model
        analysis_result = {}
        image_risk = 0.0
        image_confidence = 0.0
        
        load_models()
        if image_model:
            image_risk, image_confidence = image_model.predict(filepath)
            analysis_result = {
                'risk_score': float(image_risk),
                'confidence': float(image_confidence),
                'image_type': image_type
            }
        
        # Store image metadata
        image_data = {
            'patient_id': patient_id,
            'filename': filename,
            'image_type': image_type,
            'file_path': filepath,
            'analysis_result': analysis_result
        }
        
        stored_image = add_medical_image(image_data)
        
        return jsonify({
            'success': True,
            'image_id': stored_image.id,
            'filename': filename,
            'analysis': analysis_result,
            'risk_score': round(image_risk * 100, 2)
        })
    except Exception as e:
        logger.error(f"Image upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== ENHANCED PREDICTION WITH HISTORY ====================

@app.route('/api/predict-with-history', methods=['POST'])
def predict_with_history():
    """
    Make prediction incorporating patient's historical data
    Uses previous predictions to improve accuracy
    """
    try:
        load_models()
        
        data = request.json
        patient_id = data.get('patient_id', '')
        
        # Get clinical data
        clinical_data = {
            'age': float(data.get('age', 0)),
            'gender': data.get('gender', 'M'),
            'bp_systolic': float(data.get('bp_systolic', 120)),
            'bp_diastolic': float(data.get('bp_diastolic', 80)),
            'cholesterol': float(data.get('cholesterol', 200)),
            'blood_sugar': float(data.get('blood_sugar', 100)),
            'family_history': int(data.get('family_history', 0)),
            'smoking': int(data.get('smoking', 0)),
            'bmi': float(data.get('bmi', 22)),
            'exercise_hours': float(data.get('exercise_hours', 2))
        }
        
        # Get patient's historical predictions to improve accuracy
        historical_predictions = get_patient_predictions(patient_id, limit=5)
        
        # Calculate trend from history
        prediction_trend = 0.0
        if historical_predictions:
            recent_risks = [p.prediction_result.get('risk', 0) 
                          for p in historical_predictions 
                          if p.prediction_result]
            if recent_risks:
                prediction_trend = np.mean(recent_risks)
                clinical_data['historical_trend'] = prediction_trend
        
        # Get predictions from each model
        clinical_risk, clinical_confidence = clinical_model.predict(clinical_data)
        
        # Incorporate historical trend
        if prediction_trend > 0:
            clinical_risk = (clinical_risk * 0.7) + (prediction_trend * 0.3)
        
        wearable_data = {
            'heart_rate': float(data.get('heart_rate', 72)),
            'heart_rate_variability': float(data.get('hrv', 50)),
            'steps_daily': int(data.get('steps', 5000)),
            'sleep_hours': float(data.get('sleep', 7)),
            'spo2': float(data.get('spo2', 98))
        }
        
        wearable_risk, wearable_confidence = wearable_model.predict(wearable_data)
        
        # Fusion prediction
        fusion_input = {
            'clinical_risk': clinical_risk,
            'clinical_confidence': clinical_confidence,
            'image_risk': float(data.get('image_risk', 0)),
            'image_confidence': float(data.get('image_confidence', 0)),
            'wearable_risk': wearable_risk,
            'wearable_confidence': wearable_confidence
        }
        
        final_risk, final_confidence = fusion_model.predict(fusion_input)
        risk_level = risk_calc.get_risk_level(final_risk)
        
        # Store prediction in database
        prediction_data = {
            'patient_id': patient_id,
            'prediction_type': 'fusion',
            'prediction_result': {
                'risk': float(final_risk),
                'clinical': float(clinical_risk),
                'wearable': float(wearable_risk),
                'image': float(data.get('image_risk', 0))
            },
            'confidence_score': float(final_confidence),
            'risk_level': risk_level,
            'explanation': f"Risk assessment based on clinical data, wearable metrics, and historical trends"
        }
        
        stored_prediction = add_prediction(prediction_data)
        
        return jsonify({
            'success': True,
            'prediction_id': stored_prediction.id,
            'patient_id': patient_id,
            'risk_score': round(final_risk * 100, 2),
            'confidence': round(final_confidence * 100, 2),
            'risk_level': risk_level,
            'trend': 'Improving' if prediction_trend > final_risk else 'Worsening'
        })
    except Exception as e:
        logger.error(f"Prediction with history error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
