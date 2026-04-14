class ExplainableAI:

    def __init__(self):
        print("Explainable AI Module Initialized")

    def explain(self, clinical_data, final_risk):
        """
        Lightweight, rule-based explanations for demo use.
        Replace with SHAP or LIME later.
        """
        features = []

        if clinical_data.get("age", 0) >= 55:
            features.append("Age")
        if clinical_data.get("smoking", 0) > 0:
            features.append("Smoking Status")
        if clinical_data.get("bmi", 0) >= 30:
            features.append("BMI")
        if clinical_data.get("bp_systolic", 0) >= 140:
            features.append("Blood Pressure")
        if clinical_data.get("cholesterol", 0) >= 240:
            features.append("Cholesterol")
        if clinical_data.get("blood_sugar", 0) >= 126:
            features.append("Blood Sugar")
        if clinical_data.get("family_history", 0) == 1:
            features.append("Family History")

        if not features:
            features = ["General Health Factors"]

        if final_risk < 0.3:
            message = "Overall risk is low; maintain healthy habits."
        elif final_risk < 0.7:
            message = "Moderate risk detected; consider lifestyle improvements."
        else:
            message = "High risk detected; seek medical consultation."

        return {
            "important_features": features[:5],
            "message": message
        }

    def generate_explanation(self, prediction_result):
        """
        Legacy API retained for backward compatibility.
        """
        return {
            "important_features": ["Age", "Family History"],
            "message": "Generated using legacy explanation method."
        }
