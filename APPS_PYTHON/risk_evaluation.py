import pandas as pd

def evaluate_risk(df):
    """
    Evalúa el riesgo de churn de cada cliente aplicando reglas numéricas y categóricas,
    ajustando la probabilidad del modelo y calculando un score final.
    """
    df = df.copy()
    df.fillna("No", inplace=True)

    df["Churn_Probability_Adjusted"] = df["Churn_Probability"].apply(lambda x: round(x / 2, 2))
 
    df["Numeric_Score"] = (
        (df["tenure"].astype(float) < 12).astype(float) * 0.08 +
        (df["MonthlyCharges"].astype(float) > 70).astype(float) * 0.08 +
        (df["TotalCharges"].astype(float) < 1000).astype(float) * 0.08
    ).round(2)
    df["Categorical_Score"] = (
       ((df["Partner_Yes"] == 0) & (df["Dependents_Yes"] == 0)).astype(float) * 0.06 +  
       ((df["Contract_One year"] == 0) & (df["Contract_Two year"] == 0)).astype(float) * 0.06 +  # Mes a mes
       (df["InternetService_Fiber optic"] == 1).astype(float) * 0.06 +
       (df["TechSupport_Yes"] == 0).astype(float) * 0.06
    ).round(2)
    df["Final_Risk_Score"] = (
        df["Churn_Probability_Adjusted"] +
        df["Numeric_Score"] +
        df["Categorical_Score"]
    ).round(2)
    
    return df