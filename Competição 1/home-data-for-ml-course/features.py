import pandas as pd
import os



def carregar_treino():
    treino = pd.read_csv(os.path.join("train.csv"), sep=",", header=0, encoding="utf-8")
    df_treino = pd.DataFrame()
    df_treino["Id"] = treino["Id"]
    df_treino["TotalSF"] = treino["GrLivArea"] + treino["TotalBsmtSF"]
    df_treino["Total_Bath"] = treino["FullBath"] + (0.5 * treino["HalfBath"]) + treino["BsmtFullBath"] + (0.5 * treino["BsmtHalfBath"])
    df_treino["HouseAge"] = treino["YrSold"] - treino["YearBuilt"]
    df_treino["RemodelAge"] = treino["YrSold"] - treino["YearRemodAdd"]
    df_treino["IsNew"] = (treino["YrSold"] == treino["YearBuilt"]).astype(int)
    df_treino["ExtQuality"] = treino["ExterQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_treino["ExtCondition"] = treino["ExterCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_treino["BsmtQuality"] = treino["BsmtQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_treino["BsmtCondition"] = treino["BsmtCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_treino["HeatingQuality"] = treino["HeatingQC"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})  
    df_treino["KitchenQuality"] = treino["KitchenQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_treino["FireplaceQuality"] = treino["FireplaceQu"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_treino["GarageQuality"] = treino["GarageQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_treino["GarageCondition"] = treino["GarageCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_treino["PoolQuality"] = treino["PoolQC"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_treino["HasPool"] = treino["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    df_treino["HasFireplace"] = treino["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)
    df_treino["Has2andGarage"] = treino["MiscFeature"].apply(lambda x: 1 if x == "2ndGarage" else 0)
    df_treino["HasBasement"] = treino["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
    df_treino["OverallQual_LivArea"] = treino["OverallQual"] * treino["GrLivArea"]
    df_treino["SalePrice"] = treino["SalePrice"]

    return df_treino

def carregar_teste():
    teste = pd.read_csv(os.path.join("test.csv"), sep=",", header=0, encoding="utf-8")
    df_teste = pd.DataFrame()
    df_teste["Id"] = teste["Id"]
    df_teste["TotalSF"] = teste["GrLivArea"] + teste["TotalBsmtSF"]
    df_teste["Total_Bath"] = teste["FullBath"] + (0.5 * teste["HalfBath"]) + teste["BsmtFullBath"] + (0.5 * teste["BsmtHalfBath"])
    df_teste["HouseAge"] = teste["YrSold"] - teste["YearBuilt"]
    df_teste["RemodelAge"] = teste["YrSold"] - teste["YearRemodAdd"]
    df_teste["IsNew"] = (teste["YrSold"] == teste["YearBuilt"]).astype(int)
    df_teste["ExtQuality"] = teste["ExterQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_teste["ExtCondition"] = teste["ExterCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_teste["BsmtQuality"] = teste["BsmtQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_teste["BsmtCondition"] = teste["BsmtCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_teste["HeatingQuality"] = teste["HeatingQC"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})  
    df_teste["KitchenQuality"] = teste["KitchenQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_teste["FireplaceQuality"] = teste["FireplaceQu"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_teste["GarageQuality"] = teste["GarageQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_teste["GarageCondition"] = teste["GarageCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_teste["PoolQuality"] = teste["PoolQC"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})
    df_teste["HasPool"] = teste["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    df_teste["HasFireplace"] = teste["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)
    df_teste["Has2andGarage"] = teste["MiscFeature"].apply(lambda x: 1 if x == "2ndGarage" else 0)
    df_teste["HasBasement"] = teste["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
    df_teste["OverallQual_LivArea"] = teste["OverallQual"] * teste["GrLivArea"]
    
    return df_teste