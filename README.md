# MedAI Fusion

MedAI Fusion is a Flask-based healthcare screening platform that combines clinical inputs, medical image analysis, and wearable data into a single AI-assisted risk assessment flow.

Live demo: [multifusion-mr3a4x7ie-vijayrgaligali-3801s-projects.vercel.app](https://multifusion-mr3a4x7ie-vijayrgaligali-3801s-projects.vercel.app)

## About The Project

This project is built as an early-risk healthcare screening application. It helps users enter health-related details, upload a medical image, provide wearable metrics, and receive a combined risk score with explanations and recommendations.

The main goal is to make health screening more accessible by combining multiple health signals in one place instead of depending on only one data source.

## What The Project Does

MedAI Fusion collects three categories of input:

- Clinical data
- Medical image input
- Wearable and lifestyle data

After that, the app:

- Computes individual risk estimates
- Combines them into a final fused risk score
- Shows confidence level
- Explains important factors
- Suggests recommendations
- Displays estimated cost savings from early detection

## How It Works

### Step 1. Clinical Data

The user enters values such as:

- Age
- Gender
- Blood pressure
- Cholesterol
- Blood sugar
- BMI
- Family history
- Smoking status
- Exercise hours

This information is used to calculate a clinical risk score.

### Step 2. Medical Image Upload

The user can upload a medical image. The app processes the image and estimates risk contribution using image-based analysis.

### Step 3. Wearable Data

The user provides:

- Heart rate
- HRV
- Daily steps
- Sleep hours
- SpO2

These signals are used to calculate a wearable-based health risk score.

### Step 4. Fusion

The backend combines:

- Clinical risk
- Image risk
- Wearable risk

It then generates:

- Final risk percentage
- Confidence percentage
- Risk level
- Recommendations
- Explanation summary

## Important Features

- Multi-step health assessment flow
- Multi-modal fusion of clinical, image, and wearable inputs
- Final result page with risk, confidence, and recommendations
- Explanation layer that highlights the most important factors
- Multilingual interface support for English, Kannada, and Hindi
- Deployment-ready Flask app running on Vercel
- Input validation for all critical numeric user fields

## Input Validation

This project includes validation limits for important health fields.

Example:

- Maximum age is `100`
- If a user enters `123`, the app shows:
  `Age limit exceeded. Maximum allowed is 100.`

Similar limits are applied to:

- Systolic blood pressure
- Diastolic blood pressure
- Cholesterol
- Blood sugar
- BMI
- Exercise hours
- Heart rate
- HRV
- Steps
- Sleep
- SpO2

The validation is implemented in both:

- Frontend JavaScript for immediate user feedback
- Backend Flask logic for secure enforcement

## Pages In The Application

- `/`
  Landing page introducing the platform and its value.

- `/dashboard`
  Main assessment flow where users enter all required health details.

- `/predict`
  Backend endpoint that validates the data, runs prediction logic, and stores results.

- `/results`
  Result page that shows final risk, confidence, key factors, recommendations, and cost savings.

- `/about`
  Project mission and social impact page.

## Tech Stack

- Backend: Flask
- Frontend: HTML, CSS, JavaScript
- Data Processing: NumPy, Pandas
- Clinical Logic: scikit-learn
- Image Processing: Pillow
- Deployment: Vercel

## Project Structure

```text
mediafusion-ai/
|-- app.py
|-- requirements.txt
|-- vercel.json
|-- templates/
|-- static/
|-- models/
|-- utils/
|-- data/
|-- database/
|-- docs/
|-- translations.py
```

## Local Setup

```bash
git clone https://github.com/vijayrgali26/multifusion-ai.git
cd multifusion-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the app at:

```text
http://127.0.0.1:5000
```

## Deployment Notes

The project includes:

- `vercel.json` for Vercel routing
- `.python-version` for Python version selection
- `requirements.txt` for Python dependencies

## Why This Project Matters

MedAI Fusion is built around the idea of affordable early screening. The project aims to:

- Bring multiple health indicators into one workflow
- Improve awareness through simple AI-assisted results
- Support accessibility through multilingual UI
- Encourage earlier action before conditions become more serious

## Disclaimer

This project is a prototype and educational healthcare application. It does not replace professional medical diagnosis, treatment, or clinical decision-making.
