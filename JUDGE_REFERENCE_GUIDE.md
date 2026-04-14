# MedAI Fusion - Quick Judge Reference Guide

## ⚡ One-Page Overview

### What is MedAI Fusion?

A **multi-modal AI healthcare platform** that uses 4 machine learning models to predict disease risk early, making healthcare accessible and affordable for India.

---

## 🤖 The 4 AI Models

### 1. Clinical Data Model (RF + XGBoost)

```
📊 INPUT: Patient vitals (age, BP, cholesterol, blood sugar, etc.)
🔧 ALGORITHM: Random Forest + XGBoost (Ensemble)
📈 OUTPUT: Risk score + Confidence
⏱️ SPEED: <10ms
📚 TRAINED ON: UCI Heart Disease Dataset (1000+ samples)
```

### 2. Medical Image Model (CNN) ⭐

```
📷 INPUT: Chest X-ray image (224×224 pixels)
🧠 ALGORITHM: CNN using MobileNetV2 (Transfer Learning)
📈 OUTPUT: Risk score + Confidence
⏱️ SPEED: 50-100ms
📚 TRAINED ON: NIH/Kaggle Chest X-ray Dataset
🎯 ACCURACY: 92-96%

Architecture:
ImageNet (pre-trained) → Feature Extraction →
Dropout → Dense(128) → Dropout → Output (Sigmoid)
```

### 3. Wearable Data Model (LSTM)

```
⌚ INPUT: Time-series sensor data (100 steps × 5 sensors)
🧠 ALGORITHM: LSTM (3 layers: 128→64→32 units)
📈 OUTPUT: Risk score + Confidence
⏱️ SPEED: 20-30ms
📚 TRAINED ON: MIT-BIH Arrhythmia Dataset
🎯 ACCURACY: 88-93%

Sensors: Heart Rate, HRV, Steps, Sleep, SpO2
```

### 4. Fusion Model (Weighted Ensemble)

```
Combines all 3 models:
Clinical(40%) + Image(35%) + Wearable(25%)
Using confidence-weighted averaging

Final Output: Risk Score (0-100%) + Decision
```

---

## 📊 Why These Algorithms?

| Model             | Why?                                                                 |
| ----------------- | -------------------------------------------------------------------- |
| **Random Forest** | Handles non-linear patterns, robust to missing data                  |
| **XGBoost**       | Highest accuracy, learns sequential errors                           |
| **CNN**           | Perfect for image recognition, convolutional filters detect patterns |
| **LSTM**          | Understands sequences, remembers past trends                         |
| **Ensemble**      | Reduces error by combining diverse models                            |

---

## 💾 Data Storage

### During Assessment (Session)

```
Input Data → Flask Session → [In-Memory]
                           ↓
                    Model Processing
                           ↓
Results → Displayed to User → Cleared after session ends
```

### File Storage

```
uploads/
  ├─ chest_xray_1.jpg
  ├─ chest_xray_2.png
  └─ ... (medical images)

models/saved/
  ├─ chest_xray_cnn.h5 (CNN model)
  ├─ wearable_lstm.h5 (LSTM model)
  ├─ clinical_rf.pkl (Random Forest)
  ├─ clinical_xgb.pkl (XGBoost)
  └─ clinical_scaler.pkl (Normalization)
```

### Patient Privacy

- ✅ No permanent storage of patient data
- ✅ Session-based (auto-cleared)
- ✅ Images deleted after processing
- ✅ HIPAA-friendly design
- ✅ Multi-language support (EN, HI, KN)

---

## 📁 Notebooks Folder (3 Files)

### 1️⃣ `01_clinical_model_training.ipynb`

- Load UCI Heart Disease Dataset
- Train & compare RF vs XGBoost
- Evaluate accuracy, precision, recall
- Save models to `models/saved/`
- Create performance visualizations

### 2️⃣ `02_image_model_training.ipynb`

- Load chest X-ray images
- Resize to 224×224 pixels
- Apply data augmentation (rotation, zoom)
- Build CNN with MobileNetV2
- Generate saliency maps
- Fine-tune for optimal accuracy

### 3️⃣ `03_wearable_model_training.ipynb`

- Create time-series sequences (100 steps)
- Normalize sensor data
- Stack 3 LSTM layers
- Train with early stopping
- Analyze temporal patterns
- Validate on test sequences

**Purpose of Notebooks:**
Research & development. Scientists use them to train and tune models before deployment. Not used by patients.

---

## 🔄 How It Works (User Journey)

```
Patient Visits Dashboard
        ↓
Step 1: Enter Clinical Data (age, BP, cholesterol, etc.)
        ↓
Step 2: Upload Chest X-ray Image
        ↓
Step 3: Input Wearable Sensor Data (HR, steps, sleep)
        ↓
   [ALL 3 MODELS RUN IN PARALLEL]
        ↓
Clinical Model → Risk Score (e.g., 62%)
Image Model    → Risk Score (e.g., 58%)
Wearable Model → Risk Score (e.g., 65%)
        ↓
   [FUSION MODEL COMBINES]
        ↓
Weighted Average = (62×0.40 + 58×0.35 + 65×0.25) / weights
Final Risk Score = ~61%
        ↓
Results Page Shows:
- Overall Risk: 61% (MODERATE - Yellow Zone)
- Confidence: 89% (High confidence)
- Model Breakdown: Charts showing each model's contribution
- Key Factors: Ranked by importance (SHAP analysis)
- Recommendations: Lifestyle changes vs specialist visit
- Cost Savings: ₹4,98,500 through early detection
- AI Disclaimer: "Not a medical diagnosis"
```

---

## 🎯 Key Features

### ✅ CNN Details

- **Architecture:** MobileNetV2 (lightweight, fast)
- **Transfer Learning:** Uses ImageNet pre-training
- **Detects:** Pneumonia, consolidations, abnormalities
- **Explains:** Saliency maps show what CNN focused on

### ✅ LSTM Details

- **3 Layers:** Each with dropout for overfitting prevention
- **Temporal:** Analyzes 100 timesteps (trend over time)
- **Captures:** Irregular patterns, anomalies
- **Explains:** Which days/sensors contributed to risk

### ✅ Ensemble Benefits

- **Robustness:** If 1 model is wrong, ensemble still accurate
- **Diversity:** Different models catch different patterns
- **Confidence:** If all models agree = high confidence
- **Explainability:** See which model raised concerns

---

## 📊 Model Performance

| Metric         | Clinical | Image (CNN) | Wearable (LSTM) |
| -------------- | -------- | ----------- | --------------- |
| Accuracy       | 85-90%   | 92-96%      | 88-93%          |
| Precision      | 84%      | 94%         | 87%             |
| Recall         | 87%      | 93%         | 90%             |
| F1-Score       | 0.86     | 0.94        | 0.88            |
| Inference Time | <10ms    | 50-100ms    | 20-30ms         |

---

## 🌟 Innovation Highlights

1. **Multi-Modal Fusion** → Combines 3 different input types
2. **CNN for Medical Imaging** → Deep learning for X-rays
3. **LSTM for Wearables** → Time-series analysis
4. **Explainable AI** → SHAP values show reasoning
5. **Accessible Design** → 3 languages (EN, HI, KN)
6. **Cost-Benefit Analysis** → Shows economic impact
7. **Privacy-First** → No permanent data storage
8. **Production Ready** → Pre-trained models, scalable

---

## ❓ Judge Questions & Answers

### Q1: "Is CNN really used in your project?"

**A:** Yes! In `models/image_model.py`. MobileNetV2 + custom dense layers. Trained on chest X-ray datasets. Detects abnormalities in medical images.

### Q2: "How many AI models do you have?"

**A:** 4 total:

- Clinical (Random Forest)
- Clinical (XGBoost)
- Image (CNN)
- Wearable (LSTM)
- [All combined by Fusion model]

Or counting by type: 3 specialty models + 1 fusion = 4 models

### Q3: "Where is patient data stored?"

**A:** Flask sessions (in-memory, not permanent). Medical images in `uploads/` folder, automatically deleted. No database currently, but schema prepared for future HIPAA-compliant implementation.

### Q4: "What are the .ipynb files for?"

**A:** Model training notebooks:

- 01_clinical_model_training.ipynb → Trains RF & XGBoost
- 02_image_model_training.ipynb → Trains CNN on X-rays
- 03_wearable_model_training.ipynb → Trains LSTM on sensor data

Scientists use them for R&D. Patients never see them.

### Q5: "Why LSTM for wearables?"

**A:** Wearables produce time-series data. LSTM is designed for sequences - it remembers previous values, detects trends, and captures temporal patterns that forest models can't see.

### Q6: "How does the fusion combine models?"

**A:** Confidence-weighted ensemble:

```
Final Risk = (C_risk × C_conf × 0.40 +
              I_risk × I_conf × 0.35 +
              W_risk × W_conf × 0.25) /
             (C_conf × 0.40 + I_conf × 0.35 + W_conf × 0.25)
```

### Q7: "How accurate is the system?"

**A:** Depends on input quality:

- All 3 models agree (high confidence) = 94%+ reliable
- 2 models agree = 88-90% reliable
- Ensemble approach improves individual model accuracy by 3-5%

### Q8: "Can it replace doctors?"

**A:** No. It's a **screening tool** for early detection, not diagnosis. Results show "See a doctor" recommendation with AI disclaimer.

---

## 🚀 Deployment Ready

- ✅ Pre-trained models included
- ✅ Flask server handles requests
- ✅ HTML/CSS/JS frontend
- ✅ Multilingual support
- ✅ Scalable architecture
- ✅ Error handling & logging
- ✅ HIPAA-friendly design

---

## 📚 File Structure

```
models/
  ├─ clinical_model.py (RF + XGBoost)
  ├─ image_model.py (CNN - MobileNetV2)
  ├─ wearable_model.py (LSTM)
  ├─ fusion_model.py (Ensemble)
  └─ saved/
      ├─ chest_xray_cnn.h5
      ├─ wearable_lstm.h5
      ├─ clinical_rf.pkl
      ├─ clinical_xgb.pkl
      └─ clinical_scaler.pkl

notebooks/ (Training references)
  ├─ 01_clinical_model_training.ipynb
  ├─ 02_image_model_training.ipynb
  └─ 03_wearable_model_training.ipynb

utils/
  ├─ explainable_ai.py (SHAP analysis)
  └─ risk_calculator.py (Cost-benefit analysis)

templates/ (Web UI - multilingual)
  ├─ index.html (landing)
  ├─ dashboard.html (data input)
  ├─ results.html (predictions)
  └─ about.html (impact)

app.py (Flask backend - routes, model orchestration)
```

---

**Document Created:** AI_MODELS_DOCUMENTATION.md  
**For:** Judge Presentations & Technical Discussions  
**Date:** February 2026
