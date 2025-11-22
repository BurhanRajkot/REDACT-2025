import json
import os

# Cache encoder in memory
_cached_encoder = None

def decode_disease(encoded_value):
    global _cached_encoder

    if _cached_encoder is None:
        # Path to REDACT-2025/Data/encoder.json
        BACKEND_DIR = os.path.dirname(__file__)           # BackEnd/
        ROOT_DIR = os.path.dirname(BACKEND_DIR)           # REDACT-2025/
        ENCODER_PATH = os.path.join(ROOT_DIR, "Data", "encoder.json")

        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"encoder.json not found at {ENCODER_PATH}")

        with open(ENCODER_PATH, "r") as f:
            _cached_encoder = json.load(f)

    disease_map = _cached_encoder.get("Disease", {})

    # Build reverse mapping: 0 → "Anemia", etc.
    reverse_map = {v: k for k, v in disease_map.items()}

    return reverse_map.get(encoded_value, "Unknown Disease")
