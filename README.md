# 📊 Explainable Customer Churn Predictor

**Author:** Uhesh Dammalapati  
Built with ❤️ using Python, Streamlit, SHAP, and XGBoost.

---

## 🔍 Overview

This project predicts telecom customer churn and explains each prediction using **SHAP values**.  
An interactive **Streamlit dashboard** lets you:

- Upload a CSV of customer records
- Enter a single customer's details using a form
- Predict churn probability for each customer
- View global SHAP bar chart
- Click any row to see a SHAP waterfall plot explaining **why** that customer is at risk

---

## 🗂 Files in This Project

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit dashboard |
| `model.pkl` | Trained XGBoost model (ROC AUC ≈ 0.82) |
| `scaler.pkl` | StandardScaler used for numeric features |
| `features.pkl` | List of feature columns used during training |
| `sample_input.csv` | Sample test input for bulk predictions |
| `requirements.txt` | List of required Python packages |
| `training_notebook.ipynb` | Jupyter notebook for EDA, preprocessing, modeling, and SHAP |
| `README.md` | You're reading it! |

---

## 🧪 How to Run the App Locally

```bash
# 1. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit app
streamlit run app.py
