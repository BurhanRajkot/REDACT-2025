import os
from BackEnd import load_model, load_medical_ranges, apply_scaling

print("🔍 Loading model...")
model = load_model()
print("🔍 Loading medical ranges...")
ranges = load_medical_ranges()

# -----------------------
# SAMPLE TEST INPUT
# Modify ANY values below
# -----------------------

sample_input = {
    "Glucose": 120,
    "Cholesterol": 180,
    "Hemoglobin": 15,
    "Platelets": 250000,
    "White Blood Cells": 8000,
    "Red Blood Cells": 5.0,
    "Hematocrit": 45,
    "Mean Corpuscular Volume": 90,
    "Mean Corpuscular Hemoglobin": 30,
    "Mean Corpuscular Hemoglobin Concentration": 34,
    "Insulin": 15,
    "BMI": 22,
    "Systolic Blood Pressure": 118,
    "Diastolic Blood Pressure": 75,
    "Triglycerides": 120,
    "HbA1c": 5.2,
    "LDL Cholesterol": 110,
    "HDL Cholesterol": 52,
    "ALT": 28,
    "AST": 29,
    "Heart Rate": 80,
    "Creatinine": 1.0,
    "Troponin": 0.01,
    "C-reactive Protein": 1.2
}

print("\n📊 Scaling input...")
scaled = apply_scaling(sample_input)
print("Scaled Input:", scaled)

print("\n🤖 Running prediction...")
prediction = model.predict(scaled)[0]

try:
    proba = model.predict_proba(scaled)[0]
except:
    proba = None

print("\n======================")
print("MODEL OUTPUT")
print("======================")
print(f"Predicted Class: {prediction}")

if proba is not None:
    print("\nClass Probabilities:")
    for idx, p in enumerate(proba):
        print(f"  Class {idx}: {p*100:.2f}%")

print("\n🎉 Model test complete!")
