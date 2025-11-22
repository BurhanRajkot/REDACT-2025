import streamlit as st
import json
import os

# ------ PAGE CONFIG ------
st.set_page_config(
    page_title="MediGuard AI",
    page_icon="🩺",
    layout="wide"
)

# ------ LOAD CSS ------
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("Styles/custom.css")

# ------ FIXED: LOGO PATH ------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "Assets", "logo.png")

st.sidebar.image(logo_path, width=120)
st.sidebar.title("MediGuard AI")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Input Form", "Prediction", "Risk Indicators"]
)

# ------ HOME PAGE ------
if page == "Home":
    st.title("🩺 MediGuard AI — Intelligent Triage Assistant")
    st.markdown("""
    Welcome to **MediGuard AI**, a clinical triage tool that predicts disease likelihood
    from 24 blood parameters and provides risk explainability.

    Use the sidebar to navigate between:
    - **Input Form** → Enter patient blood values
    - **Prediction** → Get disease prediction
    - **Risk Indicators** → See what features contributed
    """)

# ------ INPUT FORM PAGE ------
elif page == "Input Form":
    st.title("📥 Input Clinical Values")
    st.write("Enter the raw values for each blood test parameter.")

    # Load scaling ranges
    with open(os.path.join(BASE_DIR, "Data", "scaler.json")) as f:
        scaling_ranges = json.load(f)

    cols = st.columns(3)
    user_inputs = {}

    i = 0
    for feature, rng in scaling_ranges.items():
        col = cols[i % 3]
        user_inputs[feature] = col.number_input(
            feature,
            min_value=float(rng[0]),
            max_value=float(rng[1]),
            step=0.01
        )
        i += 1

    st.session_state["inputs"] = user_inputs
    st.success("Input values saved! Use the Prediction page next.")

# ------ PREDICTION PAGE ------
elif page == "Prediction":
    st.title("🔮 Prediction Output")
    st.write("Model prediction will appear here.")
    st.info("Prediction logic will be connected after the ML model arrives.")

# ------ RISK INDICATOR PAGE ------
elif page == "Risk Indicators":
    st.title("📊 Risk Indicators / Explainability")
    st.write("Feature importance and explanations will appear here.")
