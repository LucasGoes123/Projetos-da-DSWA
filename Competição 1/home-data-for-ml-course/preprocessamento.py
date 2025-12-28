import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np

def imputar_valores(df, n_neighbors=5, imputer=None):
    colunas = df.columns
    if imputer is None:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputado = pd.DataFrame(imputer.fit_transform(df), columns=colunas)
        return df_imputado, imputer
    else:  
        df_imputado = pd.DataFrame(imputer.transform(df), columns=colunas)
        return df_imputado

def padronizar_dados(df, scaler=None):
    colunas = df.columns
    if scaler is None: 
        scaler = StandardScaler()
        df_padronizado = pd.DataFrame(scaler.fit_transform(df), columns=colunas)
        return df_padronizado, scaler
    else: 
        df_padronizado = pd.DataFrame(scaler.transform(df), columns=colunas)
        return df_padronizado

def processar_outliers_iqr(df):
    df_out = df.copy()
    colunas_numericas = df_out.select_dtypes(include=[np.number]).columns
    
    for coluna in colunas_numericas:
        Q1 = df_out[coluna].quantile(0.25)
        Q3 = df_out[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        df_out[coluna] = df_out[coluna].clip(lower=limite_inferior, upper=limite_superior)
    
    return df_out

def transformar_log(df, colunas):
    df_transformado = df.copy()
    for coluna in colunas:
        df_transformado[coluna] = np.log1p(df_transformado[coluna])
    return df_transformado

def preprocessar_dados(df, colunas_log=[], n_neighbors=5, is_treino=True, scaler_treino=None, imputer_treino=None):
    df_processado = df.copy()
    
    df_processado = transformar_log(df_processado, colunas_log)
    df_processado = processar_outliers_iqr(df_processado)
    
    if is_treino:
        df_processado, imputer = imputar_valores(df_processado, n_neighbors=n_neighbors)
    else:
        df_processado = imputar_valores(df_processado, imputer=imputer_treino)
    
    if is_treino:
        df_processado, scaler = padronizar_dados(df_processado)
        return df_processado, scaler, imputer
    else:
        df_processado = padronizar_dados(df_processado, scaler=scaler_treino)
        return df_processado