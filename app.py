import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

st.title("🏡 Prédiction de prix automatique")

# Charger les données
df = pd.read_excel("train_light.xlsx")

# Séparer features / target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Colonnes numériques / catégorielles
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Préprocessing
num_pipeline = SimpleImputer(strategy="median")
cat_pipeline = Pipeline([ 
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("encoder", OneHotEncoder(handle_unknown="ignore")) 
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Pipeline complet
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# Entraîner le modèle
pipeline.fit(X, y)

# Formulaire Streamlit pour ajuster les valeurs
st.sidebar.header("Ajustez les caractéristiques de la maison")

lot_area = st.sidebar.number_input("Surface du terrain (LotArea)", min_value=0, max_value=100000, value=7000, step=100)
year_built = st.sidebar.number_input("Année de construction", min_value=1800, max_value=2025, value=2005, step=1)
heating = st.sidebar.selectbox("Type de chauffage", options=["GasA", "GasW", "Grav", "Wall", "Floor", "Steam", "Hot water", "Other"], index=0)
bedrooms_above_gr = st.sidebar.number_input("Chambres au-dessus du sol (BedroomAbvGr)", min_value=0, max_value=10, value=3, step=1)
pool_area = st.sidebar.number_input("Surface de la piscine (PoolArea)", min_value=0, max_value=1000, value=0, step=10)
garage_cars = st.sidebar.number_input("Nombre de voitures dans le garage (GarageCars)", min_value=0, max_value=10, value=2, step=1)
fireplaces = st.sidebar.number_input("Nombre de cheminées (Fireplaces)", min_value=0, max_value=5, value=1, step=1)
kitchen_above_gr = st.sidebar.number_input("Cuisine au-dessus du sol (KitchenAbvGr)", min_value=0, max_value=10, value=1, step=1)
full_bath = st.sidebar.number_input("Nombre de salles de bains complètes (FullBath)", min_value=0, max_value=5, value=2, step=1)

# Créer le DataFrame à partir des entrées utilisateur
user_input = pd.DataFrame([{
    "LotArea": lot_area,
    "YearBuilt": year_built,
    "Heating": heating,
    "BedroomAbvGr": bedrooms_above_gr,
    "PoolArea": pool_area,
    "GarageCars": garage_cars,
    "Fireplaces": fireplaces,
    "KitchenAbvGr": kitchen_above_gr,
    "FullBath": full_bath
}])

# Prédiction
predicted_price = pipeline.predict(user_input)[0]

# Afficher la prédiction
st.success(f"💰 Prix estimé : {int(predicted_price):,} $")
