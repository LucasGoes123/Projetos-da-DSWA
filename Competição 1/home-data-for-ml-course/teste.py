from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

parametros = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

def testar(df_treino):

    X = df_treino.drop(columns=["Id", "SalePrice"])
    y = df_treino["SalePrice"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=parametros,
        n_iter=50,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    
    score = random_search.score(X_val, y_val)
    print(f"Score no conjunto de validação: {score:.4f}")
    print("Melhores hiperparâmetros encontrados:")
    print(random_search.best_params_)
    return random_search 