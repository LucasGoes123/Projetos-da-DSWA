import pandas as pd
import os
import numpy as np

def carregar_treino():
    treino = pd.read_csv(os.path.join('train.csv'), sep=",", header=0, encoding="utf-8")
    df_treino = pd.DataFrame()
    df_treino['ano'] = pd.to_datetime(treino['data']).dt.year
    df_treino['mes'] = pd.to_datetime(treino['data']).dt.month
    df_treino['dia'] = pd.to_datetime(treino['data']).dt.day
    df_treino['mes_sin'] = np.sin(2 * np.pi * df_treino['mes']/12)
    df_treino['mes_cos'] = np.cos(2 * np.pi * df_treino['mes']/12)
    df_treino['sku'] = treino['sku']
    df_treino['cod_filial'] = treino['cod_filial']
    df_treino['filial'] = treino['filial'].map({'RUA': 1, 'SHOPPING': 2})
    df_treino['unidade'] = treino['unidade'].map({'KG': 1, 'UN': 2})
    df_treino['is_weekend'] = pd.to_datetime(treino['data']).dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    df_treino['trimestre'] = pd.to_datetime(treino['data']).dt.quarter
    df_treino['demanda'] = treino['demanda']
    return df_treino

def carregar_teste():
    teste = pd.read_csv(os.path.join('test.csv'), sep=",", header=0, encoding="utf-8")
    df_teste = pd.DataFrame()
    df_teste['id'] = teste['id']
    df_teste['ano'] = pd.to_datetime(teste['data']).dt.year
    df_teste['mes'] = pd.to_datetime(teste['data']).dt.month
    df_teste['dia'] = pd.to_datetime(teste['data']).dt.day
    df_teste['mes_sin'] = np.sin(2 * np.pi * df_teste['mes']/12)
    df_teste['mes_cos'] = np.cos(2 * np.pi * df_teste['mes']/12)
    df_teste['sku'] = teste['sku']
    df_teste['cod_filial'] = teste['cod_filial']
    df_teste['filial'] = teste['filial'].map({'RUA': 1, 'SHOPPING': 2})
    df_teste['unidade'] = teste['unidade'].map({'KG': 1, 'UN': 2})
    df_teste['is_weekend'] = pd.to_datetime(teste['data']).dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    df_teste['trimestre'] = pd.to_datetime(teste['data']).dt.quarter
    return df_teste