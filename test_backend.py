print("🔍 Testing BackEnd package...")

try:
    from BackEnd import load_model, load_medical_ranges, apply_scaling
    print("✅ Import successful!")
except Exception as e:
    print("❌ Import FAILED:", e)
    raise SystemExit()

# Test JSON loading
print("\n--- Testing medical_ranges.json ---")
try:
    ranges = load_medical_ranges()
    print("✅ Loaded ranges successfully!")
    print("Sample keys:", list(ranges.keys())[:5])
except Exception as e:
    print("❌ Failed to load ranges:", e)

# Test model loading
print("\n--- Testing ML model file ---")
try:
    model = load_model()
    print("✅ Model loaded successfully!")
    print("Model type:", type(model))
except Exception as e:
    print("❌ Failed to load model:", e)

# Test scaling bridge (basic check)
print("\n--- Testing apply_scaling() ---")
try:
    sample_input = {
        "Glucose": 100,
        "Cholesterol": 150,
        "Hemoglobin": 15,
        "Platelets": 200000,
        "White Blood Cells": 7000,
        "Red Blood Cells": 5.0,
        "Hematocrit": 45,
        "Mean Corpuscular Volume": 90,
        "Mean Corpuscular Hemoglobin": 30,
        "Mean Corpuscular Hemoglobin Concentration": 34,
        "Insulin": 15,
        "BMI": 23,
        "Systolic Blood Pressure": 115,
        "Diastolic Blood Pressure": 75,
        "Triglycerides": 120,
        "HbA1c": 5,
        "LDL Cholesterol": 100,
        "HDL Cholesterol": 50,
        "ALT": 25,
        "AST": 25,
        "Heart Rate": 80,
        "Creatinine": 1.0,
        "Troponin": 0.02,
        "C-reactive Protein": 1.0
    }

    scaled = apply_scaling(sample_input)
    print("✅ Scaling function works!")
    print("Scaled Output:", scaled)
except Exception as e:
    print("⚠️ Scaling test skipped or failed:", e)

print("\n🎉 ALL TESTS COMPLETE")
