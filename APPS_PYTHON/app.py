import streamlit as st
import pandas as pd
import joblib
import os
from risk_evaluation import evaluate_risk


model_path = os.path.join(os.getcwd(), "random_forest_churn.pkl")
columns_path = os.path.join(os.getcwd(), "train_columns.pkl")

model = joblib.load(model_path)
train_columns = joblib.load(columns_path) 

st.title("🔍 Predicción de Abandono en Clientes")
st.write("Ingrese los datos del cliente para predecir la probabilidad de que este abandone sus servicios.")

tenure = st.number_input("Tenure (meses)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges (USD)", min_value=0.0, max_value=150.0, value=50.0)
total_charges = st.number_input("Total Charges (USD)", min_value=0.0, max_value=10000.0, value=500.0)
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])


if st.button("🔮 Predicciones de abandono"):
    
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Partner": [partner],
        "Dependents": [dependents],
        "Contract": [contract],
        "InternetService": [internet_service],
        "TechSupport": [tech_support],
        "MultipleLines": [multiple_lines],
        "DeviceProtection": [device_protection]
    })

  
    expected_categories = {
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "TechSupport": ["Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "DeviceProtection": ["No", "Yes", "No internet service"]
    }

    
    for col, categories in expected_categories.items():
        for cat in categories:
            col_name = f"{col}_{cat}"
            if col_name not in input_data.columns:
                input_data[col_name] = (input_data[col] == cat).astype(int)
    input_data.drop(columns=expected_categories.keys(), inplace=True, errors='ignore')

    
    for col in train_columns:
        if col not in input_data.columns:
            input_data[col] = 0  
    
    
    input_data = input_data.reindex(columns=train_columns, fill_value=0)
    
    # Generador de probabilidad al 100% nota: recuerda que estas diviendo entre 2 en la funcion de risk
    probability = model.predict_proba(input_data)[:, 1]  

    input_data["Churn_Probability"] = probability

    
    processed_data = evaluate_risk(input_data)

    # Muestrame el resultado
    st.subheader("📊 Resultado de la Predicción")
    st.metric(label="Probabilidad de Abandono (%)", value=round(processed_data["Final_Risk_Score"].values[0] * 100, 2))