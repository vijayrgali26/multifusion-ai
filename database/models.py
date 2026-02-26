"""
Database models for patient data and medical records
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = "sqlite:///./medai_fusion.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PatientRecord(Base):
    """Model for storing patient medical records"""
    __tablename__ = "patient_records"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True)
    name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(10))
    email = Column(String(100))
    phone = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Medical history
    medical_history = Column(JSON)  # Structured medical history
    medications = Column(JSON)  # Current medications
    allergies = Column(JSON)  # Known allergies
    lifestyle_factors = Column(JSON)  # Smoking, alcohol, exercise, diet
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'email': self.email,
            'phone': self.phone,
            'medical_history': self.medical_history,
            'medications': self.medications,
            'allergies': self.allergies,
            'lifestyle_factors': self.lifestyle_factors
        }


class PredictionRecord(Base):
    """Model for storing prediction results with history"""
    __tablename__ = "prediction_records"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), index=True)
    prediction_type = Column(String(50))  # 'clinical', 'image', 'wearable', 'fusion'
    prediction_result = Column(JSON)  # Structured prediction data
    confidence_score = Column(Float)
    risk_level = Column(String(20))  # Low, Medium, High
    explanation = Column(Text)  # AI explanation of prediction
    input_data = Column(JSON)  # Input parameters for reproducibility
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'prediction_type': self.prediction_type,
            'prediction_result': self.prediction_result,
            'confidence_score': self.confidence_score,
            'risk_level': self.risk_level,
            'explanation': self.explanation,
            'created_at': self.created_at.isoformat()
        }


class MedicalImage(Base):
    """Model for storing medical images metadata"""
    __tablename__ = "medical_images"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), index=True)
    filename = Column(String(255))
    image_type = Column(String(50))  # 'chest_xray', 'ct_scan', etc.
    file_path = Column(String(255))
    analysis_result = Column(JSON)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'filename': self.filename,
            'image_type': self.image_type,
            'analysis_result': self.analysis_result,
            'uploaded_at': self.uploaded_at.isoformat()
        }


# Create all tables
Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_patient(patient_data: dict):
    """Add a new patient record"""
    db = SessionLocal()
    try:
        patient = PatientRecord(**patient_data)
        db.add(patient)
        db.commit()
        db.refresh(patient)
        return patient
    finally:
        db.close()


def get_patient(patient_id: str):
    """Get patient record by ID"""
    db = SessionLocal()
    try:
        return db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    finally:
        db.close()


def get_patient_predictions(patient_id: str, limit: int = 10):
    """Get prediction history for a patient"""
    db = SessionLocal()
    try:
        return db.query(PredictionRecord).filter(
            PredictionRecord.patient_id == patient_id
        ).order_by(PredictionRecord.created_at.desc()).limit(limit).all()
    finally:
        db.close()


def add_prediction(prediction_data: dict):
    """Add a prediction record"""
    db = SessionLocal()
    try:
        prediction = PredictionRecord(**prediction_data)
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction
    finally:
        db.close()


def add_medical_image(image_data: dict):
    """Add medical image metadata"""
    db = SessionLocal()
    try:
        image = MedicalImage(**image_data)
        db.add(image)
        db.commit()
        db.refresh(image)
        return image
    finally:
        db.close()
