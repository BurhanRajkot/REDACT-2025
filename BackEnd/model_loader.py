import os
import json
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # goes up to REDACT-2025
DATA_DIR = os.path.join(BASE_DIR, "Data")

MODEL_PATH = os.path.join(DATA_DIR, "blood_disease_classifier_model.pkl")
RANGES_PATH = os.path.join(DATA_DIR, "medical_ranges.json")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_medical_ranges():
    if not os.path.exists(RANGES_PATH):
        raise FileNotFoundError(f"Ranges file missing: {RANGES_PATH}")
    with open(RANGES_PATH, "r") as f:
        return json.load(f)
