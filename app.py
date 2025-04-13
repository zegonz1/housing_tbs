import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# === Configuration de la page ===
st.set_page_config(
    page_title="Estimation de prix immobilier",
    page_icon="🏡",
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🏡 Estimation de prix immobilier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ajustez les caractéristiques de votre bien, puis validez pour obtenir une estimation</p>", unsafe_allow_html=True)
st.markdown("---")

# === Chargement des données ===
df = pd.read_excel("train_light.xlsx")
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# === Préparation du pipeline ===
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

num_pipeline = SimpleImputer(strategy="median")
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

pipeline.fit(X, y)

# === Interface utilisateur ===
st.sidebar.header("🔧 Caractéristiques de la maison")

col1, col2 = st.sidebar.columns(2)

with col1:
    lot_area = st.number_input("🏡 Surface du terrain (m²)", 0, 100000, 7000, 100)
    bedrooms_above_gr = st.number_input("🛏️ Chambres", 0, 10, 3)
    full_bath = st.number_input("🛁 Salles de bains", 0, 5, 2)
    fireplaces = st.number_input("🔥 Cheminées", 0, 5, 1)
    kitchen_above_gr = st.number_input("🍽️ Cuisines", 0, 5, 1)

with col2:
    year_built = st.number_input("📅 Année de construction", 1800, 2025, 2005)
    pool_area = st.number_input("🏊 Surface de piscine (m²)", 0, 1000, 0, 10)
    garage_cars = st.number_input("🚗 Garage (voitures)", 0, 10, 2)
    heating = st.selectbox("🔥 Type de chauffage", ["GasA", "GasW", "Grav", "Wall", "Floor", "Steam", "Hot water", "Other"])

# === Bouton de validation ===
if st.sidebar.button("✅ Valider les caractéristiques"):

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

    predicted_price = pipeline.predict(user_input)[0]

    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #2E86C1;'>💰 Prix estimé de la maison :</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #27AE60;'>{int(predicted_price):,} $</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Estimation générée à partir des données saisies.</p>", unsafe_allow_html=True)
