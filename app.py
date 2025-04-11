import streamlit as st
import joblib
import os
import pandas as pd
from utils.logger import get_logger
from utils.preprocessing import preprocess_data

logger = get_logger(__name__)

st.title("Loan Eligibility Prediction")
st.write("Enter the details to predict loan eligibility.")

# Load the model
model_path = os.path.join("models", "model.pkl")
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
    logger.info("Model loaded from %s", model_path)
except Exception as e:
    st.error("Failed to load the model. Please run the training script first.")
    logger.error("Error loading model: %s", e)
    st.stop()

# Input fields for all required features
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0, value=360)
credit_history = st.selectbox("Credit History", ["1", "0"])  # as strings so label encoder works
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    try:
        # Build a single-row DataFrame for the user's inputs
        input_data = {
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_amount_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        }
        df_input = pd.DataFrame(input_data)
        
        # Add a dummy target column so that preprocess_data finds it.
        if "Loan_Approved" not in df_input.columns:
            df_input["Loan_Approved"] = "Y"
        
        # Preprocess the input data (this applies label encoding for categorical columns)
        X_input, _ = preprocess_data(df_input)
        
        # Make prediction
        prediction = model.predict(X_input)
        
        st.write("Predicted Loan Approval:", "Y" if prediction[0] == 1 else "N")
        logger.info("Prediction made with input: %s", input_data)
    except Exception as e:
        st.error("Error during prediction.")
        logger.error("Prediction error: %s", e)

#test1
