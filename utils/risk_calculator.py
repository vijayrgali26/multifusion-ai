class RiskCalculator:

    def __init__(self):
        print("Risk Calculator Initialized")

    def get_risk_level(self, risk_score):
        if risk_score >= 0.7:
            return "High Risk"
        if risk_score >= 0.4:
            return "Moderate Risk"
        return "Low Risk"

    def get_recommendations(self, risk_score, clinical_data):
        recommendations = []

        if clinical_data.get("smoking", 0) > 0:
            recommendations.append("Consider a smoking cessation plan.")
        if clinical_data.get("bmi", 0) >= 25:
            recommendations.append("Aim for a healthy BMI with diet and activity.")
        if clinical_data.get("bp_systolic", 0) >= 140 or clinical_data.get("bp_diastolic", 0) >= 90:
            recommendations.append("Monitor blood pressure and reduce sodium intake.")
        if clinical_data.get("cholesterol", 0) >= 240:
            recommendations.append("Review lipid profile and reduce saturated fats.")
        if clinical_data.get("blood_sugar", 0) >= 126:
            recommendations.append("Check HbA1c and limit refined carbohydrates.")
        if clinical_data.get("exercise_hours", 0) < 2:
            recommendations.append("Increase weekly exercise to at least 150 minutes.")

        if not recommendations:
            recommendations.append("Maintain current healthy lifestyle.")

        if risk_score >= 0.7:
            recommendations.insert(0, "Schedule a clinician review soon.")
        elif risk_score >= 0.4:
            recommendations.insert(0, "Consider a follow-up screening in 3-6 months.")

        return recommendations

    def calculate_trends(self, patient_id):
        # Demo trend data; replace with database-backed history.
        return {
            "patient_id": patient_id,
            "risk_scores": [0.35, 0.4, 0.38, 0.45, 0.5],
            "timestamps": [
                "2025-10-01",
                "2025-11-01",
                "2025-12-01",
                "2026-01-01",
                "2026-02-01"
            ]
        }

    def calculate_risk(self, patient_data):
        """
        Legacy API retained for backward compatibility.
        """
        risk_score = 0.65
        return {
            "risk_score": risk_score,
            "risk_level": self.get_risk_level(risk_score)
        }
