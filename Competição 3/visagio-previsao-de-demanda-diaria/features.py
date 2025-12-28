import pandas as pd
import os
import numpy as np

def carregar_treino():
    treino = pd.read_csv(os.path.join('train.csv'), sep=",", header=0, encoding="utf-8")
    df_treino = pd.DataFrame()
  
    data_dt = pd.to_datetime(treino['data'])
    df_treino['ano'] = data_dt.dt.year
    df_treino['dia'] = data_dt.dt.day
    df_treino['mes_sin'] = np.sin(2 * np.pi * data_dt.dt.month / 12)
    df_treino['mes_cos'] = np.cos(2 * np.pi * data_dt.dt.month / 12)
    df_treino['dia_semana_sin'] = np.sin(2 * np.pi * data_dt.dt.dayofweek / 7)
    df_treino['dia_semana_cos'] = np.cos(2 * np.pi * data_dt.dt.dayofweek / 7)
    df_treino['sku'] = treino['sku']
    df_treino['cod_filial'] = treino['cod_filial']
    df_treino['unidade'] = treino['unidade'].map({'KG': 1, 'UN': 2})
    df_treino['demanda'] = treino['demanda']


    return df_treino.select_dtypes(include=[np.number])

def carregar_teste():
    teste = pd.read_csv(os.path.join('test.csv'), sep=",", header=0, encoding="utf-8")
    df_teste = pd.DataFrame()
    df_teste['id'] = teste['id'] 
    data_dt = pd.to_datetime(teste['data'])
    df_teste['ano'] = data_dt.dt.year
    df_teste['dia'] = data_dt.dt.day
    df_teste['mes_sin'] = np.sin(2 * np.pi * data_dt.dt.month / 12)
    df_teste['mes_cos'] = np.cos(2 * np.pi * data_dt.dt.month / 12)
    df_teste['dia_semana_sin'] = np.sin(2 * np.pi * data_dt.dt.dayofweek / 7)
    df_teste['dia_semana_cos'] = np.cos(2 * np.pi * data_dt.dt.dayofweek / 7)
    df_teste['sku'] = teste['sku']
    df_teste['cod_filial'] = teste['cod_filial']
    df_teste['unidade'] = teste['unidade'].map({'KG': 1, 'UN': 2})

    return df_teste.select_dtypes(include=[np.number])