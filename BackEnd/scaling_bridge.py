import json
import numpy as np

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