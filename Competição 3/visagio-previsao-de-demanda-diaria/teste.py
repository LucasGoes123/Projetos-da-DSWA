from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Mantendo seu dicionário de busca
parametros = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 10, 15],
}

def testar(df_treino):
    X = df_treino.drop(columns=["demanda"])
    y = df_treino["demanda"]
    
    
    print("Iniciando RFECV para selecionar as melhores features...")
    selector_estimator = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    rfecv = RFECV(
        estimator=selector_estimator,
        min_features_to_select=9,
        step=1,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=1
    )
    rfecv.fit(X, y)
    
    features_selecionadas = X.columns[rfecv.support_].tolist()
    print(f"Número de features reduzido de {X.shape[1]} para {len(features_selecionadas)}")
    print(f"Features mantidas: {features_selecionadas}")
    
    X_otimizado = X[features_selecionadas]
    
    X_train, X_val, y_train, y_val = train_test_split(X_otimizado, y, test_size=0.2, random_state=42)

    print("Iniciando RandomizedSearchCV nas melhores features...")
    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(learning_rate=0.05, n_jobs=1),
        param_distributions=parametros,
        n_iter=50,
        cv=tscv,
        random_state=42,
        n_jobs=1
    )

    random_search.fit(X_train, y_train)

    score = random_search.score(X_val, y_val)
    print("-" * 30)
    print(f"Score R² na validação: {score:.4f}")
    print(f"Melhores parâmetros: {random_search.best_params_}")
    
    return random_search, features_selecionadas