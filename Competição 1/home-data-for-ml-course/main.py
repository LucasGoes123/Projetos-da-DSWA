import features
import preprocessamento
import teste
import pandas as pd
import os   

def gerar_submissao(modelo_treinado, df_teste_processado):
    X_teste = df_teste_processado.drop(columns=["Id"])
    preds = modelo_treinado.predict(X_teste)
    
    submissao = pd.read_csv("sample_submission.csv")
    submissao = pd.DataFrame({
        "Id": df_teste_processado["Id"],
        "SalePrice": preds
    })
    submissao.to_csv("sample_submission.csv", index=False)
    print("Ficheiro de submissão gerado com sucesso!")

def main():
    df_treino = features.carregar_treino()
    df_teste = features.carregar_teste()
    colunas_log = ["TotalSF", "OverallQual_LivArea"]

    df_treino_processado, scaler, imputer = preprocessamento.preprocessar_dados(
        df_treino.drop(columns=["Id", "SalePrice"]),
        colunas_log=colunas_log,
        is_treino=True
    )
    
    df_treino_processado["Id"] = df_treino["Id"].values
    df_treino_processado["SalePrice"] = df_treino["SalePrice"].values

    df_teste_processado = preprocessamento.preprocessar_dados(
        df_teste.drop(columns=["Id"]),
        colunas_log=colunas_log,
        is_treino=False,
        scaler_treino=scaler,    
        imputer_treino=imputer  
    )
    
    df_teste_processado["Id"] = df_teste["Id"].values
    print("Dados de teste processados:")
    print(df_teste_processado.head())
    random_search = teste.testar(df_treino_processado)
    random_search.fit(
        df_treino_processado.drop(columns=["Id", "SalePrice"]),
        df_treino_processado["SalePrice"]
    )
    gerar_submissao(random_search, df_teste_processado)

if __name__ == "__main__":
    main()
