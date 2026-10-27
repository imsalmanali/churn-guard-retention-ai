"""
app.py
Interactive Streamlit Dashboard for ChurnGuard AI:
- Single Customer Real-Time Risk Profiler
- Retention Recommendation Engine
- Model Performance Metrics & Visualizations
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="ChurnGuard AI — Retention Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #1B365D;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metadata():
    pipeline_path = "artifacts/churn_model_pipeline.pkl"
    metadata_path = "artifacts/metadata.pkl"
    if not os.path.exists(pipeline_path):
        return None, None
    pipe = joblib.load(pipeline_path)
    meta = joblib.load(metadata_path)
    return pipe, meta

pipeline, metadata = load_model_and_metadata()

st.markdown('<div class="main-header">🛡️ ChurnGuard AI: Customer Retention Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production ML system predicting subscription churn risk and generating proactive retention strategies.</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ Model artifacts not found. Please run `python train.py` first to generate model weights.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Risk Predictor", "📊 Model Analytics & Reports", "ℹ️ About Architecture"])

with tab1:
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("📋 Customer Profile Input")
        with st.form("customer_form"):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)

            with c2:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

            c3, c4 = st.columns(2)
            with c3:
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
            with c4:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=75.0, step=1.0)
                total_charges = st.number_input("Total Lifetime Charges ($)", min_value=18.0, max_value=9000.0, value=float(monthly_charges * tenure), step=10.0)

            submitted = st.form_submit_button("🔍 Calculate Churn Risk", use_container_width=True)

    with col2:
        st.subheader("📈 Risk Assessment & Action Plan")

        if submitted:
            # Prepare feature frame
            avg_charges = total_charges / (tenure + 1e-5)
            is_new = int(tenure <= 6)

            input_df = pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "AvgChargesPerMonth": avg_charges,
                "IsNewCustomer": is_new
            }])

            prob = pipeline.predict_proba(input_df)[0][1]
            churn_risk_pct = prob * 100

            # UI Status Pill
            if churn_risk_pct >= 65:
                st.error(f"### High Churn Risk: **{churn_risk_pct:.1f}%**")
                risk_level = "CRITICAL"
            elif churn_risk_pct >= 40:
                st.warning(f"### Moderate Churn Risk: **{churn_risk_pct:.1f}%**")
                risk_level = "MODERATE"
            else:
                st.success(f"### Low Churn Risk: **{churn_risk_pct:.1f}%**")
                risk_level = "HEALTHY"

            st.progress(min(prob, 1.0))

            st.markdown("#### 💡 Prescriptive Retention Recommendations:")
            recommendations = []
            if contract == "Month-to-month":
                recommendations.append("Offer a 15% discount incentive to upgrade to a **1-Year Annual Contract**.")
            if internet_service == "Fiber optic" and tech_support != "Yes":
                recommendations.append("Bundle complimentary **Tech Support** for 3 months to resolve service friction.")
            if payment_method == "Electronic check":
                recommendations.append("Incentivize migration to **Automated Credit Card / Bank Transfer** with a $10 bill credit.")
            if tenure <= 6:
                recommendations.append("Enroll customer in the **New User Concierge Onboarding** lifecycle email sequence.")
            if not recommendations:
                recommendations.append("Customer is highly engaged and healthy. Recommended for upsell or loyalty VIP program.")

            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.info("👈 Fill out the customer parameters on the left and click **'Calculate Churn Risk'** to view the live ML inference.")

with tab2:
    st.subheader("📊 Production Model Metrics")
    m = metadata["metrics"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC Score", f"{m['roc_auc']:.4f}")
    k2.metric("Test Accuracy", f"{m['accuracy'] * 100:.1f}%")
    k3.metric("F1-Score", f"{m['f1']:.4f}")
    k4.metric("Recall (Sensitivity)", f"{m['recall'] * 100:.1f}%")

    st.markdown("---")
    st.subheader("🖼️ Validation Reports")
    r1, r2 = st.columns(2)
    if os.path.exists("reports/confusion_matrix.png"):
        r1.image("reports/confusion_matrix.png", caption="Confusion Matrix on Unseen Test Set", use_container_width=True)
    if os.path.exists("reports/model_comparison.png"):
        r2.image("reports/model_comparison.png", caption="Model Benchmark Comparison across Algorithms", use_container_width=True)

with tab3:
    st.markdown("""
    ### 🏗️ Technical Architecture
    - **Preprocessing:** Pipeline with `ColumnTransformer` applying `StandardScaler` to continuous features and `OneHotEncoder` to categorical factors.
    - **Model Selection:** Evaluated Logistic Regression (class-weighted), Random Forest, and Gradient Boosting under 5-Fold Stratified Cross Validation.
    - **Optimization Criterion:** Maximized for **ROC-AUC & Recall** to ensure high sensitivity toward churners without causing excessive false alarms.
    - **Deployment:** Streamlit Dashboard with instantaneous scikit-learn pipeline inference.
    """)
