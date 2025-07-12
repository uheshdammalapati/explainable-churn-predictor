Explainable Customer Churn Predictor
===================================

Author: Uhesh Dammalapati

Overview
--------
This project predicts telecom customer churn and explains each prediction
using SHAP values.  A Streamlit dashboard lets you:

  • Upload a CSV of customers and get churn probabilities
  • Enter a single customer's details via sidebar form
  • View global feature importance (SHAP bar chart)
  • Click on any customer to see a SHAP waterfall plot
  • Download results as CSV (optional enhancement)

Files
-----
app.py            – Streamlit app
model.pkl         – Trained XGBoost model (ROC AUC ≈ 0.82)
scaler.pkl        – StandardScaler used in training
features.pkl      – List of feature columns expected by the model
sample_input.csv  – Small test file you can upload
requirements.txt  – Python packages needed
training_notebook.ipynb – Jupyter notebook for EDA, training, and SHAP

How to Run
----------
1. Create a virtual environment (recommended) and activate it.
2. Install packages:

       pip install -r requirements.txt

3. Start the dashboard:

       python -m streamlit run app.py

4. Open your browser to http://localhost:8501 if it doesn’t open automatically.
5. Upload *sample_input.csv* or fill the sidebar form to test.

Dataset
-------
Original dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv (IBM Telco)
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

License
-------
This project is released for educational and portfolio use.
