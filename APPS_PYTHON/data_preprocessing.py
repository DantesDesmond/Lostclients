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
    
    # 1️⃣ Renombrar la variable objetivo antes de la transformación
    if 'Churn' in df.columns:
        df.rename(columns={'Churn': 'Churn_Yes'}, inplace=True)
    
    # 2️⃣ Guardar la variable objetivo si está presente
    churn_column = df['Churn_Yes'] if 'Churn_Yes' in df.columns else None
    
    # 3️⃣ Eliminar columnas irrelevantes
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    
    # 4️⃣ Convertir variables categóricas en dummies (One-Hot Encoding)
    df = pd.get_dummies(df, drop_first=True)
    
    # 5️⃣ Escalar las variables numéricas
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[float, int]).columns  # Seleccionar solo numéricas
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 6️⃣ Restaurar la variable objetivo si fue eliminada
    if churn_column is not None:
        df['Churn_Yes'] = churn_column
    
    return df