import pandas as pd

def evaluate_numeric_rules(row):
    """
    Evalúa las reglas numéricas y asigna puntajes.
    """
    score = 0
    
    # Regla 1: Tenure < 12 meses → 8.3 pts
    if row['tenure'] < 12:
        score += 8.3
    
    # Regla 2: MonthlyCharges > 70 USD → 8.3 pts
    if row['MonthlyCharges'] > 70:
        score += 8.3
    
    # Regla 3: TotalCharges < 1000 USD → 8.3 pts
    if row['TotalCharges'] < 1000:
        score += 8.3
    
    return score

def evaluate_categorical_rules(row):
    """
    Evalúa las reglas categóricas y asigna puntajes.
    """
    score = 0
    
    # Regla 4: Sin pareja y sin dependientes → 6.25 pts
    if row['Partner'] == 'No' and row['Dependents'] == 'No':
        score += 6.25
    
    # Regla 5: Contrato mes a mes → 6.25 pts
    if row['Contract'] == 'Month-to-month':
        score += 6.25
    
    # Regla 6: InternetService = Fiber optic → 6.25 pts
    if row['InternetService'] == 'Fiber optic':
        score += 6.25
    
    # Regla 7: No tiene soporte técnico → 6.25 pts
    if row['TechSupport'] == 'No':
        score += 6.25
    
    return score

def evaluate_risk(df):
    """
    Evalúa el puntaje de riesgo de cada cliente basado en reglas numéricas y categóricas.
    """
    df['Numeric_Score'] = df.apply(evaluate_numeric_rules, axis=1)
    df['Categorical_Score'] = df.apply(evaluate_categorical_rules, axis=1)
    df['Total_Risk_Score'] = df['Numeric_Score'] + df['Categorical_Score']
    return df