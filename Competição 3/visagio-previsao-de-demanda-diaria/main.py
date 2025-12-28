import pandas as pd
import features
import preprocessamento
import teste
import matplotlib.pyplot as plt
import seaborn as sns

def gerar_submissao(modelo_treinado, df_teste_processado):
    X_teste = df_teste_processado.drop(columns=["id"])
    preds = modelo_treinado.predict(X_teste)
    
    submissao = pd.read_csv("sample_submission.csv")
    submissao = pd.DataFrame({
        "id": df_teste_processado["id"],
        "demanda": preds
    })
    submissao.to_csv("sample_submission.csv", index=False)
    print("Ficheiro de submissão gerado com sucesso!")


def main():
    # Carregar dados de treino e teste
    df_treino = features.carregar_treino()
    df_teste = features.carregar_teste()


    plt.figure(figsize=(10, 8))
    corr = df_treino.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mapa de calor da correlacao entre as variaveis')
    plt.show()

    # Exibir as primeiras linhas dos dataframes carregados
    print("Dados de Treino:")
    print(df_treino.head())
    
    print("\nDados de Teste:")
    print(df_teste.head())

    df_treino_processado, scaler, imputer = preprocessamento.processar_dados(
        df_treino.drop(columns=["demanda"]),
        is_treino=True
    )
    
    df_treino_processado["demanda"] = df_treino["demanda"].values

    df_teste_processado = preprocessamento.processar_dados(
        df_teste.drop(columns=["id"]),
        is_treino=False,
        scaler_treino=scaler,    
        imputer_treino=imputer  
    )

    df_teste_processado["id"] = df_teste["id"].values
    print("Dados de teste processados:")
    print(df_teste_processado.head())

    random_search = teste.testar(df_treino_processado)
    random_search.fit(
        df_treino_processado.drop(columns=["demanda"]),
        df_treino_processado["demanda"]
    )
    gerar_submissao(random_search, df_teste_processado)



if __name__ == "__main__":
    main()