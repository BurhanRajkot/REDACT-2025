import json

def decode_disease(encoded_value):
    with open("encoder.json", "r") as f:
        encoder = json.load(f)

    disease_mapping = encoder["Disease"]
    reverse_mapping = {v: k for k, v in disease_mapping.items()}

    return reverse_mapping.get(encoded_value, "Unknown Disease")