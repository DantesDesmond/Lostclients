import pandas as pd

def evaluate_risk(df):
    """
    Evalúa el riesgo de churn de cada cliente aplicando reglas numéricas y categóricas,
    ajustando la probabilidad del modelo y calculando un score final.
    """
    # Asegurar que los datos no contengan valores nulos
    df = df.copy()
    df.fillna("No", inplace=True)
    
    # Ajustar Churn_Probability dividiéndolo por 2 y redondeando a 2 decimales hacia arriba
    df["Churn_Probability_Adjusted"] = df["Churn_Probability"].apply(lambda x: round(x / 2, 2))
    
    # Evaluar reglas numéricas y sumar los puntajes totales
    df["Numeric_Score"] = (
        (df["tenure"].astype(float) < 12).astype(float) * 0.08 +
        (df["MonthlyCharges"].astype(float) > 70).astype(float) * 0.08 +
        (df["TotalCharges"].astype(float) < 1000).astype(float) * 0.08
    ).round(2)
    
    # Evaluar reglas categóricas y sumar los puntajes totales
    df["Categorical_Score"] = (
        ((df["Partner"].astype(str).str.strip() == "No") & (df["Dependents"].astype(str).str.strip() == "No")).astype(float) * 0.06 +
        (df["Contract"].astype(str).str.strip() == "Month-to-month").astype(float) * 0.06 +
        (df["InternetService"].astype(str).str.strip() == "Fiber optic").astype(float) * 0.06 +
        (df["TechSupport"].astype(str).str.strip() == "No").astype(float) * 0.06
    ).round(2)
    
    # Calcular el score total redondeado a 2 decimales
    df["Final_Risk_Score"] = (
        df["Churn_Probability_Adjusted"] +
        df["Numeric_Score"] +
        df["Categorical_Score"]
    ).round(2)
    
    return df