import json
import numpy as np
import os

class MedicalScaler:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.ranges = json.load(f)

        self.features = list(self.ranges.keys())

    def scale_value(self, raw_value, min_val, max_val):
        if raw_value < min_val:
            raw_value = min_val
        elif raw_value > max_val:
            raw_value = max_val

        return (raw_value - min_val) / (max_val - min_val)

    def transform(self, raw_dict):
        scaled = []

        for feature in self.features:
            min_val, max_val = self.ranges[feature]
            raw_value = raw_dict[feature]

            scaled_val = self.scale_value(raw_value, min_val, max_val)
            scaled.append(scaled_val)

        return np.array(scaled).reshape(1, -1)


# 🔥 IMPORTANT WRAPPER for app.py
def apply_scaling(raw_dict):
    """
    Public function used by app.py.
    Loads medical_ranges.json and returns scaled vector.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # REDACT-2025/
    JSON_PATH = os.path.join(BASE_DIR, "Data", "medical_ranges.json")

    scaler = MedicalScaler(JSON_PATH)
    return scaler.transform(raw_dict)
