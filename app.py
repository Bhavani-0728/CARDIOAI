import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
import os
import subprocess

st.set_page_config(page_title="CardioAI", layout="wide")

# -------------------- PREMIUM BLACK STYLING --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #000000, #0a0f1c);
    color: white;
}

/* Header */
.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: white;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 25px;
}

/* Section Title */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
    color: white;
}

/* Risk Badges with Glow */
.risk-badge-low {
    background-color: #064e3b;
    color: #34d399;
    padding: 10px 20px;
    border-radius: 30px;
    display: inline-block;
    font-weight: 600;
    box-shadow: 0 0 15px rgba(52, 211, 153, 0.4);
}

.risk-badge-moderate {
    background-color: #78350f;
    color: #fbbf24;
    padding: 10px 20px;
    border-radius: 30px;
    display: inline-block;
    font-weight: 600;
    box-shadow: 0 0 15px rgba(251, 191, 36, 0.4);
}

.risk-badge-high {
    background-color: #7f1d1d;
    color: #f87171;
    padding: 10px 20px;
    border-radius: 30px;
    display: inline-block;
    font-weight: 600;
    box-shadow: 0 0 15px rgba(248, 113, 113, 0.4);
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/best_model.pkl"
FEATURE_PATH = "models/feature_columns.json"


@st.cache_resource
def load_model():

    # train model if missing
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found. Training model...")
        subprocess.run(["python", "train.py"])

    # check again
    if not os.path.exists(MODEL_PATH):
        st.error("Model training failed. Model file still missing.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH) as f:
        feature_columns = json.load(f)

    return model, feature_columns


model, feature_columns = load_model()

# -------------------- HEADER --------------------
st.markdown("<div class='title'>❤️ CardioAI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Cardiovascular Risk Intelligence</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("📝 Health Details")

age = st.sidebar.slider("Age", 18, 100, 40)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
height = st.sidebar.slider("Height (cm)", 140, 210, 170)
weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)

st.sidebar.markdown("---")

ap_hi = st.sidebar.slider("Systolic BP", 80, 200, 120)
ap_lo = st.sidebar.slider("Diastolic BP", 50, 150, 80)

st.sidebar.markdown("---")

cholesterol = st.sidebar.selectbox("Cholesterol Level", [1, 2, 3])
gluc = st.sidebar.selectbox("Glucose Level", [1, 2, 3])

st.sidebar.markdown("---")

smoke = st.sidebar.selectbox("Smoker", ["No", "Yes"])
alco = st.sidebar.selectbox("Alcohol", ["No", "Yes"])
active = st.sidebar.selectbox("Physically Active", ["Yes", "No"])

analyze = st.sidebar.button("🚀 Analyze Health")

# -------------------- ENCODING --------------------
gender = 1 if gender == "Male" else 0
smoke = 1 if smoke == "Yes" else 0
alco = 1 if alco == "Yes" else 0
active = 1 if active == "Yes" else 0

# -------------------- PREDICTION --------------------
if analyze:

    BMI = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo

    input_dict = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "BMI": BMI,
        "pulse_pressure": pulse_pressure,
    }

    filtered_input = {k: input_dict[k] for k in feature_columns}
    input_df = pd.DataFrame([filtered_input])

    probability = model.predict_proba(input_df)[0][1]
    health_score = int((1 - probability) * 100)

    st.markdown("## 🧠 AI Health Report")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # -------------------- CIRCULAR GAUGE --------------------
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={
                'suffix': "/100",
                'font': {'color': "white", 'size': 42}
            },
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {
                    'color': "limegreen" if health_score > 70
                    else "orange" if health_score > 40
                    else "red"
                },
                'steps': [
                    {'range': [0, 40], 'color': '#1a0000'},
                    {'range': [40, 70], 'color': '#2d2100'},
                    {'range': [70, 100], 'color': '#002b1a'}
                ],
            }
        ))

        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------- INSIGHTS --------------------
    with col2:
        st.markdown("<div class='section-title'>🔍 Key Insights</div>", unsafe_allow_html=True)

        insights = []

        if age > 50:
            insights.append("Age increases cardiovascular vulnerability")

        if ap_hi > 140:
            insights.append("Elevated systolic blood pressure detected")

        if BMI > 30:
            insights.append("Obesity significantly impacts heart health")

        if smoke == 1:
            insights.append("Smoking elevates cardiovascular risk")

        if active == 0:
            insights.append("Increasing physical activity can reduce long-term risk")

        if insights:
            for i in insights:
                st.write(f"• {i}")
        else:
            st.write("• No major lifestyle risk signals detected")

        st.markdown("---")

        if probability < 0.40:
            st.markdown("<div class='risk-badge-low'>🟢 Low Risk</div>", unsafe_allow_html=True)
        elif probability < 0.70:
            st.markdown("<div class='risk-badge-moderate'>🟡 Moderate Risk</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-badge-high'>🔴 High Risk</div>", unsafe_allow_html=True)

    st.caption("AI-based cardiovascular estimation. Not a medical diagnosis.")