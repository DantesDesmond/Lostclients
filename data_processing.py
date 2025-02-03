import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Realiza la limpieza y transformación de datos para estandarizar el dataset.
    """

    # 1️⃣ Eliminar columnas irrelevantes
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)  # ID del cliente no aporta al modelo

    # 2️⃣ Convertir valores categóricos binarios a 0 y 1
    binary_columns = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    # 3️⃣ Convertir variables categóricas en dummies (One-Hot Encoding)
    df = pd.get_dummies(df, drop_first=True)

    # 4️⃣ Reemplazar valores vacíos en columnas numéricas con la media
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[:, col] = df[col].fillna(df[col].mean())

    # 5️⃣ Normalizar columnas numéricas
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df