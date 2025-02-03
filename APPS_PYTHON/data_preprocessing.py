import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    Realiza la transformación de datos:
    - Elimina columnas irrelevantes
    - Convierte variables categóricas en dummies
    - Escala las variables numéricas
    - Asegura que la variable objetivo 'Churn_Yes' se mantenga
    """
    df = df.copy()  # Evitar modificar el original
    
    # 1️⃣ Eliminar columnas irrelevantes
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    
    # 2️⃣ Convertir variables categóricas en dummies (One-Hot Encoding)
    df = pd.get_dummies(df, drop_first=True)
    
    # 3️⃣ Asegurar que 'Churn_Yes' no se elimine
    if 'Churn_Yes' in df.columns:
        churn_column = df['Churn_Yes']
    else:
        churn_column = None
    
    # 4️⃣ Escalar las variables numéricas
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[float, int]).columns  # Seleccionar solo numéricas
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 5️⃣ Restaurar la variable objetivo si fue eliminada
    if churn_column is not None:
        df['Churn_Yes'] = churn_column
    
    return df