"""
AI Assistant Module with Natural Language Processing and Multilingual Support
"""
from transformers import pipeline
import json
import logging

logger = logging.getLogger(__name__)


class AIAssistant:
    """
    Intelligent AI Assistant for healthcare guidance
    Supports multilingual queries and directs users to relevant sections
    """
    
    def __init__(self):
        """Initialize the AI Assistant"""
        # Lazy load NLP models - don't load until needed
        self.qa_pipeline = None
        self.sentiment_pipeline = None
        self._models_loaded = False
        
        # Module mapping for user queries
        self.module_routing = {
            'clinical': ['symptoms', 'disease', 'diagnosis', 'medical', 'health', 'condition'],
            'image_analysis': ['image', 'xray', 'scan', 'radiology', 'medical image', 'ct', 'mri'],
            'wearable': ['heart', 'pulse', 'wearable', 'step', 'activity', 'movement', 'monitoring'],
            'risk_assessment': ['risk', 'danger', 'severe', 'critical', 'prognosis', 'outcome'],
            'dashboard': ['data', 'history', 'records', 'previous', 'past'],
            'medications': ['medicine', 'drug', 'prescription', 'tablet', 'treatment', 'therapy']
        }
        
        # Context-aware responses
        self.responses = {
            'clinical': "Please provide your symptoms and medical history for clinical analysis.",
            'image_analysis': "Upload a medical image (X-ray, CT scan, MRI) for AI-powered analysis.",
            'wearable': "Connect your wearable device data for real-time health monitoring.",
            'risk_assessment': "Our AI will assess your health risk based on multiple data sources.",
            'dashboard': "View your complete medical history and previous predictions.",
            'medications': "Check your current medications and potential interactions."
        }
        
        # Language codes
        self.supported_languages = ['en', 'hi', 'ta', 'te', 'ka', 'ml', 'bn', 'mr', 'gu']
    
    def _load_models(self):
        """Lazy load NLP models on first use"""
        if self._models_loaded:
            return
        
        try:
            self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            self._models_loaded = True
            logger.info("NLP models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load transformers models: {e}. Using fallback mode.")
            self._models_loaded = True  # Mark as attempted to avoid repeated failures
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text
        Returns language code (en, hi, ta, etc.)
        """
        try:
            from langdetect import detect
            detected = detect(text)
            if detected in self.supported_languages:
                return detected
            return 'en'
        except:
            return 'en'
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text between languages
        Using free translation API as fallback
        """
        if source_lang == target_lang:
            return text
        
        try:
            from transformers import MarianMTModel, MarianTokenizer
            model_name = f'Helsinki-NLP/Opus-MT-{source_lang}-{target_lang}'
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            outputs = model.generate(**inputs)
            translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return translated[0] if translated else text
        except Exception as e:
            logger.warning(f"Translation failed: {e}. Returning original text.")
            return text
    
    def classify_intent(self, user_input: str) -> dict:
        """
        Classify user intent and route to appropriate module
        Returns module name and confidence score
        """
        user_input_lower = user_input.lower()
        intent_scores = {}
        
        # Keyword-based classification
        for module, keywords in self.module_routing.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                intent_scores[module] = score / len(keywords)
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            return {
                'intent': best_intent,
                'confidence': intent_scores[best_intent],
                'all_intents': intent_scores
            }
        
        return {
            'intent': 'dashboard',
            'confidence': 0.5,
            'all_intents': intent_scores
        }
    
    def generate_response(self, user_input: str, language: str = 'en') -> dict:
        """
        Generate contextual response based on user input
        """
        # Detect language if not provided
        detected_lang = self.detect_language(user_input)
        
        # Translate to English for processing if needed
        if detected_lang != 'en':
            processed_input = self.translate_text(user_input, detected_lang, 'en')
        else:
            processed_input = user_input
        
        # Classify intent
        intent_result = self.classify_intent(processed_input)
        module = intent_result['intent']
        
        # Generate response
        response = {
            'user_input': user_input,
            'detected_language': detected_lang,
            'intent': module,
            'confidence': intent_result['confidence'],
            'guidance': self.responses.get(module, "Please specify what you need help with."),
            'action': f'navigate_to_{module}'
        }
        
        # Translate response back to user's language
        if detected_lang != 'en' and language != detected_lang:
            response['guidance'] = self.translate_text(
                response['guidance'], 
                'en', 
                detected_lang
            )
        
        return response
    
    def answer_question(self, question: str, context: str = None) -> dict:
        """
        Answer healthcare questions using QA model
        """
        # Lazy load models on first use
        self._load_models()
        
        if not self.qa_pipeline:
            return {
                'question': question,
                'answer': "QA model not available. Please contact support.",
                'confidence': 0
            }
        
        default_context = """
        MedAI Fusion is a multi-modal AI healthcare platform that provides:
        1. Clinical prediction using patient symptoms and history
        2. Medical image analysis for chest X-rays and CT scans
        3. Wearable device integration for continuous health monitoring
        4. Risk assessment combining multiple data sources
        5. Personalized health recommendations with explainable AI
        """
        
        context_to_use = context or default_context
        
        try:
            result = self.qa_pipeline(question=question, context=context_to_use)
            return {
                'question': question,
                'answer': result['answer'],
                'confidence': result['score'],
                'start': result['start'],
                'end': result['end']
            }
        except Exception as e:
            logger.error(f"QA failed: {e}")
            return {
                'question': question,
                'answer': "Unable to answer. Please try rephrasing your question.",
                'confidence': 0
            }
    
    def get_health_guidance(self, symptoms: list, language: str = 'en') -> dict:
        """
        Provide health guidance based on symptoms
        """
        guidance = {
            'symptoms_entered': symptoms,
            'assistant_recommendation': f"Based on your symptoms {', '.join(symptoms)}, we recommend:",
            'suggested_modules': [],
            'next_steps': []
        }
        
        # Determine which modules to use
        symptom_text = ' '.join(symptoms)
        intent = self.classify_intent(symptom_text)
        
        if intent['confidence'] > 0.3:
            guidance['suggested_modules'].append(intent['intent'])
        
        # Always suggest fusion model for comprehensive analysis
        guidance['suggested_modules'].append('fusion')
        guidance['suggested_modules'] = list(set(guidance['suggested_modules']))
        
        # Provide next steps
        guidance['next_steps'] = [
            "1. Complete your medical history profile",
            "2. Upload relevant medical images if available",
            "3. Connect your wearable device for additional data",
            "4. Review personalized risk assessment",
            "5. Consult with healthcare provider for final diagnosis"
        ]
        
        # Translate if needed
        if language != 'en':
            for key in ['assistant_recommendation', 'next_steps']:
                if isinstance(guidance[key], list):
                    guidance[key] = [
                        self.translate_text(item, 'en', language) 
                        for item in guidance[key]
                    ]
                else:
                    guidance[key] = self.translate_text(guidance[key], 'en', language)
        
        return guidance


# Initialize assistant
assistant = AIAssistant()


def get_assistant_response(user_input: str, language: str = 'en'):
    """
    Main function to get assistant response
    """
    return assistant.generate_response(user_input, language)


def answer_health_question(question: str, context: str = None):
    """
    Main function to answer health questions
    """
    return assistant.answer_question(question, context)
