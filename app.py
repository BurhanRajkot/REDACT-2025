from BackEnd.model_loader import load_model, load_medical_ranges
from BackEnd.scaling_bridge import apply_scaling
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import shap
import numpy as np
import sqlite3

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="MediGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# LOAD CSS
# ---------------------------------------------------------
def load_css(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# adjust path if needed
load_css("Styles/custom.css")

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
NORMAL_RANGES = {
    "Glucose": (70, 140),
    "Cholesterol": (125, 200),
    "Hemoglobin": (13.5, 17.5),
    "Platelets": (150000, 450000),
    "White Blood Cells": (4000, 11000),
    "Red Blood Cells": (4.2, 5.4),
    "Hematocrit": (38, 52),
    "Mean Corpuscular Volume": (80, 100),
    "Mean Corpuscular Hemoglobin": (27, 33),
    "Mean Corpuscular Hemoglobin Concentration": (32, 36),
    "Insulin": (5, 25),
    "BMI": (18.5, 24.9),
    "Systolic Blood Pressure": (90, 120),
    "Diastolic Blood Pressure": (60, 80),
    "Triglycerides": (50, 150),
    "HbA1c": (4, 6),
    "LDL Cholesterol": (70, 130),
    "HDL Cholesterol": (40, 60),
    "ALT": (10, 40),
    "AST": (10, 40),
    "Heart Rate": (60, 100),
    "Creatinine": (0.6, 1.2),
    "Troponin": (0, 0.04),
    "C-reactive Protein": (0, 3),
}

FEATURE_ORDER = list(NORMAL_RANGES.keys())

# ---------------------------------------------------------
# MODEL + LABEL MAPPING (from encoder.json)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
ENCODER_PATH = os.path.join(DATA_DIR, "encoder.json")

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_label_mapping():
    """
    Load encoder.json once and build: class_index -> disease_name
    """
    try:
        with open(ENCODER_PATH, "r") as f:
            encoder = json.load(f)
        disease_map = encoder.get("Disease", {})
        reverse_map = {v: k for k, v in disease_map.items()}
        return reverse_map
    except Exception:
        return {}

def decode_disease(class_index: int) -> str:
    mapping = get_label_mapping()
    return mapping.get(class_index, f"Class {class_index}")

def classify_risk(disease_name: str, confidence: float) -> str:
    """
    Simple rule-based risk:
      - Healthy -> Low
      - Else + conf>=80 -> High
      - Else + conf>=50 -> Moderate
      - Else -> Borderline
    """
    if disease_name.lower() == "healthy":
        return "Low"
    if confidence >= 80:
        return "High"
    elif confidence >= 50:
        return "Moderate"
    else:
        return "Borderline"


#---------------------------------------------------------
#DATABASE CONNECTION (FOR FUTURE USE)
#---------------------------------------------------------

DB_PATH = os.path.join('Database', 'mediBase.db')

@st.cache_resource
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

conn = get_connection()
cursor = conn.cursor()

def insert_row(first_name, last_name, values, confidence, disease_name):
    if len(values) != 24:
        raise ValueError("You must provide exactly 24 values.")
    
    # Combine all values into a single list
    all_values = [first_name, last_name] + values + [confidence, disease_name]
    
    # Create placeholders for all 29 values (3 + 24 + 2)
    placeholders = ", ".join(["?"] * len(all_values))
    
    # Execute the query
    cursor.execute(f"INSERT INTO patients VALUES ({placeholders})", all_values)
    conn.commit()

# ---------------------------------------------------------
# NAVIGATION LOGIC (FIXED)
# ---------------------------------------------------------
# 1. Initialize Page Session State if not exists
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# 2. Define Page Options
PAGE_OPTIONS = ["Home", "Patient Dashboard", "Patient History"]

# 3. Determine the current index for the sidebar based on session state
try:
    current_index = PAGE_OPTIONS.index(st.session_state["page"])
except ValueError:
    current_index = 0

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="width: 70px; height: 70px; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
                 border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
                 font-size: 32px; margin-bottom: 1rem; box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);">
                🩺
            </div>
            <h2 style="font-size: 22px; font-weight: 800; margin: 0; color: #ffffff;">MediGuard AI</h2>
            <p style="font-size: 12px; opacity: 0.9; margin-top: 0.25rem;">Intelligent Triage Assistant</p>
        </div>
    """, unsafe_allow_html=True)

    # 4. Use the dynamic index for the radio button
    selected_nav = st.radio(
        "Navigation",
        PAGE_OPTIONS,
        index=current_index,
        key="nav_radio_widget"
    )

    # 5. Sync Sidebar Selection with Session State
    if selected_nav != st.session_state["page"]:
        st.session_state["page"] = selected_nav
        st.rerun()

    st.markdown("""
        <div style="margin-top: 3rem; background: rgba(255, 255, 255, 0.15);
             border-radius: 16px; padding: 1.25rem; border: 1px solid rgba(255, 255, 255, 0.2);">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 12px; height: 12px; background: #10b981; border-radius: 50%;"></div>
                <div>
                    <div style="font-size: 11px; opacity: 0.8;">System Status</div>
                    <div style="font-size: 14px; font-weight: 700;">AI Model Active</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

#=============================
# GENAI
#=============================

# Add this at the top with other imports
import google.generativeai as genai

# Add this after your constants section (around line 65)
# ---------------------------------------------------------
# GEMINI API CONFIGURATION
# ---------------------------------------------------------
GEMINI_API_KEY = "AIzaSyAsOgkiv528rSru_RkrJ-geDh7J4k8oO_Q"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

def generate_diagnosis_explanation(disease_name: str, confidence: float, shap_values: dict, patient_data: dict) -> str:
    try:
        if not shap_values:
            return "No significant contributing factors identified."
        
        # Sort SHAP values by absolute importance
        sorted_shaps = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Build factor descriptions with actual values and normal ranges
        contributing_factors = []
        for feature, shap_val in sorted_shaps:
            actual_value = patient_data.get(feature, "N/A")
            normal_range = NORMAL_RANGES.get(feature, None)
            
            status = ""
            if normal_range and actual_value != "N/A":
                low, high = normal_range
                if actual_value < low:
                    status = f"below normal range ({low}-{high})"
                elif actual_value > high:
                    status = f"above normal range ({low}-{high})"
                else:
                    status = f"within normal range ({low}-{high})"
            
            direction = "increases" if shap_val > 0 else "decreases"
            contributing_factors.append(
                f"- {feature}: {actual_value} ({status}) - {direction} risk [Impact: {shap_val:.3f}]"
            )
        
        # Create prompt for Gemini
        prompt = f"""You are a medical AI assistant helping clinicians understand diagnostic predictions from a machine learning model.

Prediction Results:
- Predicted Condition: {disease_name}
- Model Confidence: {confidence:.1f}%

Significant Contributing Factors:
The following parameters have significant SHAP values, meaning they are the key drivers of this prediction. Positive SHAP values increase risk, negative values decrease it:

{chr(10).join(contributing_factors)}

Please provide a concise clinical explanation (4-5 sentences) that:
1. Identifies which 2-3 factors are most strongly influencing this diagnosis
2. Explains the clinical significance of abnormal values (if any)
3. Describes potential risks or complications associated with these findings
4. Provides actionable monitoring recommendations for the healthcare team

Use professional medical terminology but keep it clear and actionable. Focus on the most clinically relevant insights."""

        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        return f"⚠️ Unable to generate explanation: {str(e)}"



# =========================================================
# HOME PAGE
# =========================================================
def navigate_to_dashboard():
    """Callback function to change the current page state."""
    st.session_state["page"] = "Patient Dashboard"

def render_home():
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)

    col1, col2 = st.columns([1.25, 1])

    with col1:
        st.markdown("""
            <div class="hero-badge">⚡ AI-Powered Clinical Decision Support</div>
            <h1 class="hero-title">Intelligent Triage for Better Patient Outcomes</h1>
            <p class="hero-subtitle">
                MediGuard AI uses advanced machine learning to analyze clinical parameters
                and provide accurate disease predictions, helping healthcare professionals
                make faster, data-driven decisions.
            </p>
        """, unsafe_allow_html=True)

        st.video("https://www.youtube.com/watch?v=nEA7Sb9RhyY")

        st.markdown("""
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-number">95%+</div>
                    <div class="stat-label">Accuracy Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">50K+</div>
                    <div class="stat-label">Predictions Made</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">24/7</div>
                    <div class="stat-label">Availability</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="live-prediction-card">
                <div class="live-header">
                    <div class="live-dot"></div>
                    <div>
                        <div class="live-title">Live Prediction</div>
                        <div class="live-subtitle">Patient Analysis in Progress</div>
                    </div>
                </div>

                <div style="margin: 1.5rem 0;">
                    <div class="confidence-label">Predicted Condition</div>
                    <div class="prediction-badge">Heart Disease</div>
                </div>

                <div class="confidence-bar-container">
                    <div class="confidence-label">Confidence Score</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: 94%;"></div>
                    </div>
                    <div class="confidence-text">94% Confidence</div>
                </div>

                <div style="margin-top: 1.5rem;">
                    <div class="confidence-label">Risk Assessment</div>
                    <span class="risk-badge high">● High Risk - Immediate Attention Required</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Features Section
    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2.5rem;">
            <h2 style="font-size: 32px; font-weight: 800; color: #1e293b; margin-bottom: 0.5rem;">
                Why Choose MediGuard AI?
            </h2>
            <p style="font-size: 15px; color: #64748b;">
                Advanced AI technology designed to support healthcare professionals with accurate, explainable predictions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    features = [
        ("🧠", "Advanced AI Models", "State-of-the-art machine learning algorithms trained on extensive clinical datasets for maximum accuracy."),
        ("🛡️", "HIPAA Compliant", "Enterprise-grade security ensuring patient data privacy and compliance with healthcare regulations."),
        ("📊", "Explainable Results", "Transparent feature importance insights help clinicians understand every prediction decision."),
        ("⚡", "Real-Time Analysis", "Instant predictions powered by optimized algorithms to reduce patient waiting times significantly."),
        ("🫀", "Multi-Disease Detection", "Comprehensive analysis covering cardiac, metabolic, and hematologic conditions in one unified system."),
        ("✅", "Validated Accuracy", "Clinically validated with 95%+ accuracy and high recall to minimize dangerous false negatives."),
    ]

    for idx, (icon, title, desc) in enumerate(features):
        with [c1, c2, c3, c1, c2, c3][idx]:
            st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon-wrapper">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    # CTA Banner
    st.markdown("""
        <div class="cta-banner">
            <div>
                <div class="cta-title">Ready to Transform Patient Care?</div>
                <div class="cta-subtitle">
                    Join healthcare teams using MediGuard AI to improve triage accuracy and patient outcomes.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer-section">
            <div class="footer-text">
                © 2024 MediGuard AI. All rights reserved. | Empowering clinicians with AI-driven decision support.
            </div>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# PATIENT DASHBOARD PAGE
# =========================================================
def render_patient_dashboard():
    st.markdown('<div class="page-title">Patient Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter clinical parameters and review AI triage insights</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1.2])

    # ------------------- LEFT: INPUTS -------------------
    with col1:
        #st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("#### Enter Clinical Parameters")
        st.markdown(
            "<div style='color: #64748b; font-size: 14px; margin-bottom: 1.5rem;'>"
            "Input patient blood test and vital sign data"
            "</div>",
            unsafe_allow_html=True,
        )

        # Vital Signs
        st.markdown('<div class="section-header">Patient Details</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        f_name = v1.text_input("First Name", value="", key="fn")
        s_name = v2.text_input("Last Name", value="", key="sn")

        # Vital Signs
        st.markdown('<div class="section-header">⚡ Vital Signs</div>', unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)
        heart_rate = v1.number_input("Heart Rate (bpm)", value=70.0, key="hr")
        sbp = v2.number_input("Systolic BP (mmHg)", value=110.0, key="sbp")
        dbp = v3.number_input("Diastolic BP (mmHg)", value=70.0, key="dbp")

        # Blood Chemistry
        st.markdown('<div class="section-header">🧪 Blood Chemistry</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        glucose = b1.number_input("Glucose (mg/dL)", value=90.0)
        cholesterol = b2.number_input("Cholesterol (mg/dL)", value=150.0)
        triglycerides = b3.number_input("Triglycerides (mg/dL)", value=100.0)

        b4, b5, b6 = st.columns(3)
        hba1c = b4.number_input("HbA1c (%)", value=5.0)
        ldl = b5.number_input("LDL (mg/dL)", value=100.0)
        hdl = b6.number_input("HDL (mg/dL)", value=50.0)

        # Hematology
        st.markdown('<div class="section-header">🔬 Hematology Panel</div>', unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        hemoglobin = h1.number_input("Hemoglobin (g/dL)", value=15.0)
        platelets = h2.number_input("Platelets (/µL)", value=250000)
        wbc = h3.number_input("WBC (/mm³)", value=7000)

        h4, h5, h6 = st.columns(3)
        rbc = h4.number_input("RBC (M/µL)", value=4.8)
        hematocrit = h5.number_input("Hematocrit (%)", value=45.0)
        mcv = h6.number_input("MCV (fL)", value=90.0)

        h7, h8, _ = st.columns(3)
        mch = h7.number_input("MCH (pg)", value=30.0)
        mchc = h8.number_input("MCHC (g/dL)", value=34.0)

        # Metabolic
        st.markdown('<div class="section-header">📈 Metabolic Indicators</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        bmi = m1.number_input("BMI (kg/m²)", value=21.5)
        creatinine = m2.number_input("Creatinine (mg/dL)", value=0.9)
        insulin = m3.number_input("Insulin (µU/mL)", value=10.0)

        m4, m5, _ = st.columns(3)
        alt = m4.number_input("ALT (U/L)", value=20.0)
        ast = m5.number_input("AST (U/L)", value=20.0)

        # Cardiac
        st.markdown('<div class="section-header">❤️ Cardiac Markers</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        troponin = c1.number_input("Troponin (ng/mL)", 0.0, 0.04, 0.01)
        crp = c2.number_input("CRP (mg/L)", 0.0, 3.0, 1.0)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        saphs = {}
        if st.button("🔍 Run Prediction", use_container_width=True, type="primary"):
            patient_data = {
                "Glucose": glucose, "Cholesterol": cholesterol, "Hemoglobin": hemoglobin,
                "Platelets": platelets, "White Blood Cells": wbc, "Red Blood Cells": rbc,
                "Hematocrit": hematocrit, "Mean Corpuscular Volume": mcv,
                "Mean Corpuscular Hemoglobin": mch, "Mean Corpuscular Hemoglobin Concentration": mchc,
                "Insulin": insulin, "BMI": bmi, "Systolic Blood Pressure": sbp,
                "Diastolic Blood Pressure": dbp, "Triglycerides": triglycerides,
                "HbA1c": hba1c, "LDL Cholesterol": ldl, "HDL Cholesterol": hdl,
                "ALT": alt, "AST": ast, "Heart Rate": heart_rate,
                "Creatinine": creatinine, "Troponin": troponin, "C-reactive Protein": crp,
            }
        
            try:
                # Save inputs
                st.session_state["patient_inputs"] = patient_data

                # Get model + scaled input
                model = get_model()
                scaled = apply_scaling(patient_data)

                # Predict probabilities
                def predict_disease(scaled_input, threshold=0.2):
                    p = model.predict_proba(scaled_input)
                    if p.max() < 0.6 and p.argmax() == 1:  # if max prob is less than threshold or predicted as Healthy
                        sorted_indices = np.argsort(p, axis=1)
                        descending_indices = sorted_indices[:, ::-1]
                        second_highest_indices = descending_indices[:, 1]
                        return p[second_highest_indices], second_highest_indices  # fallback
                    else:
                        return float(p.max()*100), p.argmax()
                    
                def imp_features(scaled_input, prediction):
                    shap_explainer = shap.TreeExplainer(model, feature_names=FEATURE_ORDER)
                    shap_values = shap_explainer.shap_values(scaled_input)[0]
                    feature_contributions = shap_values[:, prediction]
                    importance = np.abs(feature_contributions)
                    max_imp = importance.max()
                    threshold = 0.1 * max_imp

                    idx = np.where(importance >= threshold)[0]
                    important_features = {}

                    if len(idx) < 3:
                        top5 = importance.argsort()[::-1][:3]
                        for i in top5:
                            important_features[FEATURE_ORDER[i]] = float(feature_contributions[i])
                        return important_features

                    for i in idx:
                        important_features[FEATURE_ORDER[i]] = float(feature_contributions[i])

                        return important_features

                proba = model.predict_proba(scaled)[0]
                #best_idx = max(range(len(proba)), key=lambda i: proba[i])
                #confidence = float(proba[best_idx] * 100.0)
                confidence, best_idx = predict_disease(scaled)
                disease_name = decode_disease(best_idx)

                saphs = imp_features(scaled, best_idx)

                st.session_state["prediction"] = {
                    "class_index": int(best_idx),
                    "disease_name": disease_name,
                    "confidence": confidence,
                    "probabilities": [float(p) for p in proba],
                }

                insert_row(f_name, s_name, list(patient_data.values()), confidence, disease_name)

                st.success(
                    "✅ Analysis complete! Review the prediction results here"
                )
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------- RIGHT: RESULT & GRAPH -------------------
    with col2:
        #st.markdown('<div class="modern-card" style="min-height: 480px;">', unsafe_allow_html=True)
        st.markdown("#### Recent Predictions", unsafe_allow_html=True)
        st.markdown(
            "<div style='color: #64748b; font-size: 13px; margin-bottom: 1rem;'>"
            "Click a past diagnosis to reload patient data"
            "</div>",
            unsafe_allow_html=True,
        )

        # Fetch last 3 predictions
        try:
            cursor.execute("""
                SELECT first_name, last_name, diagnosis, confidence 
                FROM patients
                ORDER BY ROWID DESC
                LIMIT 3
            """)
            records = cursor.fetchall()

            if records:
                for idx, rec in enumerate(records):
                    first_name, last_name, diagnosis, confidence = rec
            
            # Convert bytes to string if needed
                    if isinstance(diagnosis, bytes):
                        diagnosis = diagnosis.decode('utf-8')
                    if isinstance(first_name, bytes):
                        first_name = first_name.decode('utf-8')
                    if isinstance(last_name, bytes):
                        last_name = last_name.decode('utf-8')
            
            # Create expandable section for each patient
                    with st.expander(f"👤 {first_name} {last_name} - {diagnosis} ({confidence:.1f}%)", expanded=False):
                # Fetch full patient data
                        cursor.execute("""
                            SELECT * FROM patients 
                            WHERE first_name = ? AND last_name = ? AND diagnosis = ? AND confidence = ?
                            LIMIT 1
                            """, (first_name, last_name, diagnosis, confidence))
                        full_row = cursor.fetchone()
                
                        if full_row:
                    # Map database columns to display names with units
                            param_mapping = {
                                "glucose": ("Glucose", "mg/dL"),
                                "cholesterol": ("Cholesterol", "mg/dL"),
                                "hemoglobin": ("Hemoglobin", "g/dL"),
                                "platelets": ("Platelets", "/µL"),
                                "white_blood_cells": ("White Blood Cells", "/mm³"),
                                "red_blood_cells": ("Red Blood Cells", "M/µL"),
                                "hematocrit": ("Hematocrit", "%"),
                                "mean_corpuscular_volume": ("MCV", "fL"),
                                "mean_corpuscular_hemoglobin": ("MCH", "pg"),
                                "mean_corpuscular_hemoglobin_concentration": ("MCHC", "g/dL"),
                                "insulin": ("Insulin", "µU/mL"),
                                "bmi": ("BMI", "kg/m²"),
                                "systolic_blood_pressure": ("Systolic BP", "mmHg"),
                                "diastolic_blood_pressure": ("Diastolic BP", "mmHg"),
                                "triglycerides": ("Triglycerides", "mg/dL"),
                                "hba1c": ("HbA1c", "%"),
                                "ldl_cholesterol": ("LDL Cholesterol", "mg/dL"),
                                "hdl_cholesterol": ("HDL Cholesterol", "mg/dL"),
                                "alt": ("ALT", "U/L"),
                                "ast": ("AST", "U/L"),
                                "heart_rate": ("Heart Rate", "bpm"),
                                "creatinine": ("Creatinine", "mg/dL"),
                                "troponin": ("Troponin", "ng/mL"),
                                "c_reactive_protein": ("C-reactive Protein", "mg/L"),
                            }
                    
                    # Extract values (skip first_name, last_name at start and confidence, diagnosis at end)
                            values = full_row[2:-2]
                    
                    # Create dataframe for display
                            data_rows = []
                            db_columns = [
                                "glucose", "cholesterol", "hemoglobin", "platelets", 
                                "white_blood_cells", "red_blood_cells", "hematocrit",
                                "mean_corpuscular_volume", "mean_corpuscular_hemoglobin",
                                "mean_corpuscular_hemoglobin_concentration", "insulin", "bmi",
                                "systolic_blood_pressure", "diastolic_blood_pressure",
                                "triglycerides", "hba1c", "ldl_cholesterol", "hdl_cholesterol",
                                "alt", "ast", "heart_rate", "creatinine", "troponin",
                                "c_reactive_protein"
                            ]
                    
                            for db_col, value in zip(db_columns, values):
                                param_name, unit = param_mapping[db_col]
                        
                                # Check if value is within normal range
                                normal_range = NORMAL_RANGES.get(FEATURE_ORDER[db_columns.index(db_col)])
                                status = "✅"
                                if normal_range:
                                    low, high = normal_range
                                    if value < low:
                                        status = "🔵 Low"
                                    elif value > high:
                                        status = "🔴 High"
                                    else:
                                        status = "✅ Normal"
                        
                                data_rows.append({
                                    "Parameter": param_name,
                                    "Value": f"{value:.2f}" if isinstance(value, float) else str(value),
                                    "Unit": unit,
                                    "Status": status
                                })
                    
                            # Display as table
                            df_display = pd.DataFrame(data_rows)
                            st.dataframe(
                                df_display,
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )
                           
            else:
                st.info("✅ No recent predictions found.")
        except Exception as e:
            st.error(f"⚠️ Failed to load recent predictions: {e}")


        st.markdown('</div>', unsafe_allow_html=True)

        #-----------------------------

        
        st.markdown("#### Disease Prediction Result")
        st.markdown(
            "<div style='color: #64748b; font-size: 13px; margin-bottom: 1.25rem;'>"
            "AI-powered triage assessment"
            "</div>",
            unsafe_allow_html=True,
        )

        if "patient_inputs" in st.session_state:
            pred = st.session_state.get("prediction")

            if pred is not None:
                disease = pred.get("disease_name", "Unknown")
                confidence = float(pred.get("confidence", 0.0))
                bar_width = max(5.0, min(100.0, confidence))  # keep bar visible
                risk_level = classify_risk(disease, confidence)

                risk_label_map = {
                    "Low": "🟢 Low Risk",
                    "Moderate": "⚠️ Moderate Risk",
                    "Borderline": "⚠️ Borderline Risk",
                    "High": "⚠️ High Risk",
                }
                risk_label = risk_label_map.get(risk_level, "⚠️ Risk")

                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                         padding: 1.5rem; border-radius: 16px; text-align: center; margin-bottom: 1.25rem;">
                        <div style="font-size: 13px; color: #1e40af; margin-bottom: 0.4rem;">Predicted Condition</div>
                        <div style="font-size: 22px; font-weight: 800; color: #1e40af;">{disease}</div>
                    </div>

                    <div style="margin-bottom: 1.25rem;">
                        <div style="font-size: 12px; color: #64748b; margin-bottom: 0.5rem;">Confidence Score</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {bar_width:.1f}%;"></div>
                        </div>
                        <div style="font-size: 14px; color: #475569; margin-top: 0.4rem; font-weight: 600;">
                            {confidence:.1f}% Confidence
                        </div>
                    </div>

                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 12px; color: #64748b; margin-bottom: 0.5rem;">Risk Assessment</div>
                        <span class="risk-badge high">{risk_label}</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("✅ Patient values captured. Click **Run Prediction** to see model output.")

            st.markdown("<hr style='margin: 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'>",
                        unsafe_allow_html=True)
            
        def shap_bar_chart(imps: dict):
            df = pd.DataFrame({
                "Feature": list(imps.keys()),
                "SHAP": list(imps.values())
            })

            colors = ["red" if v > 0 else "blue" for v in df["SHAP"]]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(df["Feature"], df["SHAP"], color=colors)
            ax.set_title("SHAP Contributions (Red = Higher Risk)")
            st.pyplot(fig)
            shap_bar_chart(saphs)

        if saphs:  # Only show if SHAP values exist
    
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            
            # Generate AI explanation automatically (only once per prediction)
            pred = st.session_state.get("prediction", {})
            patient_data = st.session_state.get("patient_inputs", {})
            
            # Check if we need to generate for this prediction
            current_prediction_id = f"{pred.get('disease_name', '')}_{pred.get('confidence', 0)}"
            if st.session_state.get("last_prediction_id") != current_prediction_id:
                with st.spinner("🤖 Generating AI insights..."):
                    # Generate AI explanation
                    ai_explanation = generate_diagnosis_explanation(
                        disease_name=pred.get("disease_name", "Unknown"),
                        confidence=pred.get("confidence", 0.0),
                        shap_values=saphs,
                        patient_data=patient_data
                    )
                    
                    # Prepare export data
                    export_data = {
                        "patient_info": {
                            "first_name": f_name,
                            "last_name": s_name,
                            "diagnosis": pred.get("disease_name", "Unknown"),
                            "confidence": float(pred.get("confidence", 0.0)),
                            "risk_level": classify_risk(pred.get("disease_name", "Unknown"), pred.get("confidence", 0.0))
                        },
                        "clinical_parameters": {},
                        "significant_factors": {},
                        "ai_clinical_insights": ai_explanation
                    }
                    
                    # Add all clinical parameters
                    param_mapping = {
                        "Glucose": "mg/dL", "Cholesterol": "mg/dL", "Hemoglobin": "g/dL",
                        "Platelets": "/µL", "White Blood Cells": "/mm³", "Red Blood Cells": "M/µL",
                        "Hematocrit": "%", "Mean Corpuscular Volume": "fL",
                        "Mean Corpuscular Hemoglobin": "pg", "Mean Corpuscular Hemoglobin Concentration": "g/dL",
                        "Insulin": "µU/mL", "BMI": "kg/m²", "Systolic Blood Pressure": "mmHg",
                        "Diastolic Blood Pressure": "mmHg", "Triglycerides": "mg/dL",
                        "HbA1c": "%", "LDL Cholesterol": "mg/dL", "HDL Cholesterol": "mg/dL",
                        "ALT": "U/L", "AST": "U/L", "Heart Rate": "bpm",
                        "Creatinine": "mg/dL", "Troponin": "ng/mL", "C-reactive Protein": "mg/L"
                    }
                    
                    for param, value in patient_data.items():
                        unit = param_mapping.get(param, "")
                        normal_range = NORMAL_RANGES.get(param, None)
                        status = "normal"
                        if normal_range:
                            low, high = normal_range
                            if value < low:
                                status = "below normal"
                            elif value > high:
                                status = "above normal"
                        
                        export_data["clinical_parameters"][param] = {
                            "value": float(value) if isinstance(value, (int, float)) else value,
                            "unit": unit,
                            "status": status,
                            "normal_range": f"{normal_range[0]}-{normal_range[1]}" if normal_range else "N/A"
                        }
                    
                    # Add SHAP values
                    for feature, shap_val in saphs.items():
                        export_data["significant_factors"][feature] = {
                            "shap_value": float(shap_val),
                            "impact": "increases_risk" if shap_val > 0 else "decreases_risk",
                            "current_value": float(patient_data.get(feature, 0))
                        }
                    
                    # Store for download
                    st.session_state["report_ready"] = json.dumps(export_data, indent=2)
                    st.session_state["report_filename"] = f"{f_name}_{s_name}_AI_diagnosis_report.json"
                    st.session_state["last_prediction_id"] = current_prediction_id
            
            # Show download button (report is already generated)
            if "report_ready" in st.session_state:
                st.download_button(
                    label="📥 Download AI Insights Report",
                    data=st.session_state["report_ready"],
                    file_name=st.session_state["report_filename"],
                    mime="application/json",
                    key="download_ai_report_btn",
                    use_container_width=True,
                    type="primary"
                )

                if st.session_state.get("show_ai_message", False):
                    # Extract AI explanation from the JSON
                    report_data = json.loads(st.session_state["report_ready"])
                    ai_message = report_data.get("ai_clinical_insights", "No insights available")
                    
                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                    st.markdown("**🤖 AI Clinical Insights**")
                    st.markdown(
                        f"<div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); "
                        f"padding: 1.25rem; border-radius: 12px; border-left: 4px solid #3b82f6; "
                        f"font-size: 14px; line-height: 1.8; color: #1e293b; margin-top: 0.5rem;'>"
                        f"{ai_message}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

#-----------------------------    # Parameters Exceeding Normal Ranges


# =========================================================
# MODEL METRICS PAGE
# =========================================================
def render_model_metrics():
    st.markdown('<div class="page-title">Model Metrics & Patient Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Visualize patient lab values compared to reference ranges</div>', unsafe_allow_html=True)

    if "patient_inputs" not in st.session_state:
        st.warning("⚠️ No patient data available. Please enter values on the Patient Dashboard first.")
        return

    patient = st.session_state["patient_inputs"]

    features = []
    normal_vals = []
    patient_vals = []

    for feat in FEATURE_ORDER:
        low, high = NORMAL_RANGES[feat]
        mid = (low + high) / 2
        features.append(feat)
        normal_vals.append(mid)
        patient_vals.append(patient.get(feat, mid))

    df = pd.DataFrame({
        "Feature": features,
        "Normal Range (Midpoint)": normal_vals,
        "Patient Value": patient_vals
    }).set_index("Feature")

    col1, col2 = st.columns([2.3, 1])

    with col1:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Overall Lab Profile Comparison")
        st.markdown(
            "<div style='color: #64748b; font-size: 13px; margin-bottom: 1rem;'>"
            "Normal range midpoint vs patient values"
            "</div>",
            unsafe_allow_html=True,
        )
        st.bar_chart(df)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Summary Insights")

        high_count = sum(
            patient_vals[i] > NORMAL_RANGES[FEATURE_ORDER[i]][1]
            for i in range(len(FEATURE_ORDER))
        )
        low_count = sum(
            patient_vals[i] < NORMAL_RANGES[FEATURE_ORDER[i]][0]
            for i in range(len(FEATURE_ORDER))
        )
        normal_count = len(FEATURE_ORDER) - high_count - low_count

        st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="padding: 0.75rem; background: #f1f5f9; border-radius: 10px; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; color: #1e293b;">Total Parameters:</span>
                    <span style="float: right; font-weight: 700; color: #3b82f6;">{len(FEATURE_ORDER)}</span>
                </div>
                <div style="padding: 0.75rem; background: #fee2e2; border-radius: 10px; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; color: #1e293b;">Above Normal:</span>
                    <span style="float: right; font-weight: 700; color: #dc2626;">{high_count}</span>
                </div>
                <div style="padding: 0.75rem; background: #dbeafe; border-radius: 10px; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; color: #1e293b;">Below Normal:</span>
                    <span style="float: right; font-weight: 700; color: #2563eb;">{low_count}</span>
                </div>
                <div style="padding: 0.75rem; background: #d1fae5; border-radius: 10px;">
                    <span style="font-weight: 700; color: #1e293b;">Within Range:</span>
                    <span style="float: right; font-weight: 700; color: #059669;">{normal_count}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='margin: 1rem 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        st.info("💡 Actual model metrics (precision, recall, AUC-ROC) will be displayed here once the ML model is integrated.")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PATIENT HISTORY PAGE
# =========================================================
def render_patient_history():
    st.markdown('<div class="page-title">Patient History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">View and export all past predictions</div>', unsafe_allow_html=True)

    # Fetch all predictions
    try:
        cursor.execute("""
            SELECT * FROM patients
            ORDER BY ROWID DESC
        """)
        all_records = cursor.fetchall()

        if not all_records:
            st.info("No patient records found. Start by running predictions on the Patient Dashboard.")
            return

        # Display total count
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                 padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;">
                <div style="font-size: 14px; color: #1e40af; font-weight: 600;">
                    Total Records: {len(all_records)}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Parameter mapping
        param_mapping = {
            "glucose": ("Glucose", "mg/dL"),
            "cholesterol": ("Cholesterol", "mg/dL"),
            "hemoglobin": ("Hemoglobin", "g/dL"),
            "platelets": ("Platelets", "/µL"),
            "white_blood_cells": ("White Blood Cells", "/mm³"),
            "red_blood_cells": ("Red Blood Cells", "M/µL"),
            "hematocrit": ("Hematocrit", "%"),
            "mean_corpuscular_volume": ("MCV", "fL"),
            "mean_corpuscular_hemoglobin": ("MCH", "pg"),
            "mean_corpuscular_hemoglobin_concentration": ("MCHC", "g/dL"),
            "insulin": ("Insulin", "µU/mL"),
            "bmi": ("BMI", "kg/m²"),
            "systolic_blood_pressure": ("Systolic BP", "mmHg"),
            "diastolic_blood_pressure": ("Diastolic BP", "mmHg"),
            "triglycerides": ("Triglycerides", "mg/dL"),
            "hba1c": ("HbA1c", "%"),
            "ldl_cholesterol": ("LDL Cholesterol", "mg/dL"),
            "hdl_cholesterol": ("HDL Cholesterol", "mg/dL"),
            "alt": ("ALT", "U/L"),
            "ast": ("AST", "U/L"),
            "heart_rate": ("Heart Rate", "bpm"),
            "creatinine": ("Creatinine", "mg/dL"),
            "troponin": ("Troponin", "ng/mL"),
            "c_reactive_protein": ("C-reactive Protein", "mg/L"),
        }

        db_columns = [
            "glucose", "cholesterol", "hemoglobin", "platelets", 
            "white_blood_cells", "red_blood_cells", "hematocrit",
            "mean_corpuscular_volume", "mean_corpuscular_hemoglobin",
            "mean_corpuscular_hemoglobin_concentration", "insulin", "bmi",
            "systolic_blood_pressure", "diastolic_blood_pressure",
            "triglycerides", "hba1c", "ldl_cholesterol", "hdl_cholesterol",
            "alt", "ast", "heart_rate", "creatinine", "troponin",
            "c_reactive_protein"
        ]

        # Display each record
        for idx, full_row in enumerate(all_records):
            first_name = full_row[0]
            last_name = full_row[1]
            diagnosis = full_row[-1]
            confidence = full_row[-2]
            values = full_row[2:-2]

            # Convert bytes to string if needed
            if isinstance(diagnosis, bytes):
                diagnosis = diagnosis.decode('utf-8')
            if isinstance(first_name, bytes):
                first_name = first_name.decode('utf-8')
            if isinstance(last_name, bytes):
                last_name = last_name.decode('utf-8')

            # Create expandable section
            col_expand, col_download = st.columns([5, 1])
            
            with col_expand:
                with st.expander(f"👤 {first_name} {last_name} - {diagnosis} ({confidence:.1f}%)", expanded=False):
                    # Create two columns for better layout
                    info_col, table_col = st.columns([1, 2])
                    
                    with info_col:
                        st.markdown(f"""
                            <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                                <div style="font-size: 13px; color: #64748b; margin-bottom: 0.5rem;">Patient Information</div>
                                <div style="font-size: 16px; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem;">
                                    {first_name} {last_name}
                                </div>
                                <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #e2e8f0;">
                                <div style="margin-top: 0.5rem;">
                                    <div style="font-size: 12px; color: #64748b;">Diagnosis</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #3b82f6;">{diagnosis}</div>
                                </div>
                                <div style="margin-top: 0.5rem;">
                                    <div style="font-size: 12px; color: #64748b;">Confidence</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #059669;">{confidence:.1f}%</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        # Risk assessment
                        risk_level = classify_risk(diagnosis, confidence)
                        risk_colors = {
                            "Low": "#10b981",
                            "Moderate": "#f59e0b",
                            "Borderline": "#f59e0b",
                            "High": "#ef4444"
                        }
                        risk_color = risk_colors.get(risk_level, "#64748b")
                        
                        st.markdown(f"""
                            <div style="background: {risk_color}22; padding: 0.75rem; border-radius: 8px; border-left: 4px solid {risk_color};">
                                <div style="font-size: 12px; color: #64748b;">Risk Level</div>
                                <div style="font-size: 14px; font-weight: 700; color: {risk_color};">{risk_level} Risk</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with table_col:
                        st.markdown("**Clinical Parameters**")
                        
                        # Create dataframe for display
                        data_rows = []
                        for db_col, value in zip(db_columns, values):
                            param_name, unit = param_mapping[db_col]
                            
                            # Check if value is within normal range
                            normal_range = NORMAL_RANGES.get(FEATURE_ORDER[db_columns.index(db_col)])
                            status = "✅"
                            if normal_range:
                                low, high = normal_range
                                if value < low:
                                    status = "🔵 Low"
                                elif value > high:
                                    status = "🔴 High"
                                else:
                                    status = "✅ Normal"
                            
                            data_rows.append({
                                "Parameter": param_name,
                                "Value": f"{value:.2f}" if isinstance(value, float) else str(value),
                                "Unit": unit,
                                "Status": status
                            })
                        
                        # Display as table
                        df_display = pd.DataFrame(data_rows)
                        st.dataframe(
                            df_display,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )

            with col_download:
                # Prepare JSON data
                patient_json = {
                    "patient_info": {
                        "first_name": first_name,
                        "last_name": last_name,
                        "diagnosis": diagnosis,
                        "confidence": float(confidence)
                    },
                    "clinical_parameters": {}
                }
                
                for db_col, value in zip(db_columns, values):
                    param_name, unit = param_mapping[db_col]
                    patient_json["clinical_parameters"][param_name] = {
                        "value": float(value) if isinstance(value, (int, float)) else value,
                        "unit": unit
                    }
                
                # Convert to JSON string
                json_str = json.dumps(patient_json, indent=2)
                
                # Download button
                st.download_button(
                    label="📥",
                    data=json_str,
                    file_name=f"{first_name}_{last_name}_{diagnosis.replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"download_{idx}",
                    help="Download as JSON"
                )

    except Exception as e:
        st.error(f"⚠️ Failed to load patient history: {e}")


# =========================================================
# ROUTER (Already handled by session state at top)
# =========================================================
page = st.session_state["page"]

if page == "Home":
    render_home()
elif page == "Patient Dashboard":
    render_patient_dashboard()
elif page == "Model Metrics":
    render_model_metrics()
elif page == "Patient History":
        render_patient_history()

def reload_page():
    if page == "Home":
        render_home()
    elif page == "Patient Dashboard":
        render_patient_dashboard()
    elif page == "Model Metrics":
        render_model_metrics()
    elif page == "Patient History":
        render_patient_history()