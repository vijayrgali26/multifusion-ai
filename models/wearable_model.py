"""
Wearable data analysis with lightweight heuristic scoring.
"""

import numpy as np


class WearablePredictor:
    def __init__(self):
        """Initialize wearable prediction model."""
        self.sequence_length = 100

    def predict(self, wearable_data):
        """Estimate risk from wearable vitals without deep learning."""
        heart_rate = wearable_data['heart_rate']
        hrv = wearable_data['heart_rate_variability']
        steps = wearable_data['steps_daily']
        sleep = wearable_data['sleep_hours']
        spo2 = wearable_data['spo2']

        heart_rate_risk = min(abs(heart_rate - 72) / 80, 1.0)
        hrv_risk = min(max((50 - hrv) / 50, 0), 1.0)
        activity_risk = min(max((6000 - steps) / 6000, 0), 1.0)
        sleep_risk = min(abs(sleep - 7.5) / 7.5, 1.0)
        spo2_risk = min(max((98 - spo2) / 10, 0), 1.0)

        final_risk = (
            0.28 * heart_rate_risk +
            0.22 * hrv_risk +
            0.16 * activity_risk +
            0.14 * sleep_risk +
            0.20 * spo2_risk
        )

        consistency = np.std([heart_rate_risk, hrv_risk, activity_risk, sleep_risk, spo2_risk])
        confidence = float(np.clip(0.90 - consistency * 0.25, 0.65, 0.92))

        return float(np.clip(final_risk, 0.0, 1.0)), confidence
