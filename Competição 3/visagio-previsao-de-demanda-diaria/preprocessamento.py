import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer # Adicionado SimpleImputer
import numpy as np

def imputar_valores(df, imputer=None):
    colunas = df.columns
    if imputer is None:
        imputer = SimpleImputer(strategy='mean') 
        df_imputado = pd.DataFrame(imputer.fit_transform(df), columns=colunas)
        return df_imputado, imputer
    else:
        df_imputado = pd.DataFrame(imputer.transform(df), columns=colunas)
        return df_imputado
    
def normalizar_dados(df, scaler=None):
    colunas = df.columns
    if scaler is None:
        scaler = StandardScaler()
        df_normalizado = pd.DataFrame(scaler.fit_transform(df), columns=colunas)
        return df_normalizado, scaler
    else:
        df_normalizado = pd.DataFrame(scaler.transform(df), columns=colunas)
        return df_normalizado
    
def processar_outlier(df):
    df_outlier = df.copy()
    colunas_num = df_outlier.select_dtypes(include=[np.number]).columns

    for coluna in colunas_num:
        Q1 = df_outlier[coluna].quantile(0.25)
        Q3 = df_outlier[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        df_outlier[coluna] = df_outlier[coluna].clip(lower=limite_inferior, upper=limite_superior)
    
    return df_outlier

def processar_dados(df, is_treino=True, scaler_treino=None, imputer_treino=None):
    df_processado = df.copy()
    
    df_processado = processar_outlier(df_processado)
    
    if is_treino:
        df_processado, imputer = imputar_valores(df_processado)
    else:
        df_processado = imputar_valores(df_processado, imputer=imputer_treino)
    
    if is_treino:
        df_processado, scaler = normalizar_dados(df_processado)
        return df_processado, scaler, imputer
    else:
        df_processado = normalizar_dados(df_processado, scaler=scaler_treino)
        return df_processado