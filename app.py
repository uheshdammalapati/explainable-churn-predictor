import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Explainable Churn Predictor", layout="wide")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    """Load model, scaler, and metadata saved during training."""
    model = joblib.load("model.pkl")            # trained XGBClassifier
    scaler = joblib.load("scaler.pkl")          # fitted StandardScaler
    feature_order = joblib.load("features.pkl") # list of columns in training order
    # Create SHAP explainer once
    explainer = shap.Explainer(model)
    return model, scaler, feature_order, explainer


def preprocess(df_raw: pd.DataFrame, scaler: StandardScaler, feature_order: list[str]) -> pd.DataFrame:
    """Replicate the preprocessing pipeline used for the model training."""
    df = df_raw.copy()

    # Encode target if present (will be ignored for inference)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Label‑encode binary categorical features
    label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    le = LabelEncoder()
    for col in label_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # One‑hot encode remaining categoricals (align columns after get_dummies)
    cat_cols_to_ohe = [col for col in df.columns if df[col].dtype == 'object' and col not in label_cols]
    df = pd.get_dummies(df, columns=cat_cols_to_ohe, drop_first=True)

    # Numerical columns scaling (match training set)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if set(num_cols).issubset(df.columns):
        df[num_cols] = scaler.transform(df[num_cols])

    # Reindex to model's feature order (fill missing with 0)
    df_aligned = df.reindex(columns=feature_order, fill_value=0)
    return df_aligned


def run_inference(df_features: pd.DataFrame, model: XGBClassifier):
    """Return predictions and probabilities."""
    probs = model.predict_proba(df_features)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
st.title("📊 Explainable Customer Churn Predictor")

st.markdown("Use this app to **predict churn** for telecom customers and **see why** the model thinks so (SHAP explanations). Upload multiple customers via CSV or enter a single customer's details.")

model, scaler, feature_order, explainer = load_artifacts()

# -----------------------------------------------------------------------------
# Sidebar input mode
# -----------------------------------------------------------------------------
mode = st.sidebar.radio("Prediction mode", ["Bulk CSV Upload", "Single Customer Form"])

if mode == "Bulk CSV Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.dataframe(df_raw.head(), use_container_width=True)
        df_proc = preprocess(df_raw, scaler, feature_order)
        preds, probs = run_inference(df_proc, model)
        df_results = df_raw.copy()
        df_results["Churn_Prob"] = probs
        df_results["Churn_Pred"] = preds
        st.success("Predictions generated!")

        # Summary stats
        churn_rate = np.mean(df_results['Churn_Pred']) * 100
        st.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")

        # Display table
        st.dataframe(df_results.sort_values('Churn_Prob', ascending=False), use_container_width=True)

        # SHAP global explanation
        st.subheader("🔍 Global Feature Importance (SHAP)")
        shap_values = explainer(df_proc)
        plt.tight_layout()
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)

elif mode == "Single Customer Form":
    st.sidebar.markdown("### Enter customer details")

    # --- minimal subset of features for demo ---
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly = st.sidebar.number_input("Monthly Charges", 10.0, 150.0, 70.0)
    total = st.sidebar.number_input("Total Charges", 10.0, 9000.0, 2500.0)
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

    if st.sidebar.button("Predict Churn"):
        # Build single‑row dataframe
        input_dict = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': 'No',
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }
        df_single = pd.DataFrame([input_dict])
        df_proc = preprocess(df_single, scaler, feature_order)
        pred, prob = run_inference(df_proc, model)

        st.write("### Prediction Result")
        st.write(f"**Churn Probability:** {prob[0]:.2%}")
        st.write("**Prediction:**", "Churn" if pred[0] == 1 else "No Churn")

        st.subheader("Why? (SHAP explanation)")
        shap_values = explainer(df_proc)
        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig2)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.caption("Built with ❤️ using Streamlit and SHAP | Author: Uhesh Dammalapati")

