# 📊 Explainable Customer Churn Predictor

**Author&nbsp;– Uhesh Dammalapati**

---

## Overview
This project predicts telecom customer churn **and explains each prediction** using SHAP values.  
A Streamlit dashboard lets you:

- **Upload** a CSV of customers and get churn probabilities  
- **Enter** a single customer's details via the sidebar form  
- **View** global feature importance (SHAP bar chart)  
- **Click** any customer to see a SHAP waterfall plot  
- **(Optional)** Download results as CSV  

---

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard |
| `model.pkl` | Trained XGBoost model (ROC AUC ≈ 0.82) |
| `scaler.pkl` | `StandardScaler` used in training |
| `features.pkl` | List of feature columns expected by the model |
| `sample_input.csv` | Small test file you can upload |
| `requirements.txt` | Python packages needed |
| `training_notebook.ipynb` | Jupyter notebook for EDA, training, SHAP |

---

## How to Run Locally
```bash
# 1️⃣ (Optional) create & activate a virtual env  
#    python -m venv venv && venv\Scripts\activate

# 2️⃣ Install dependencies  
pip install -r requirements.txt

# 3️⃣ Launch the dashboard  
python -m streamlit run app.py
