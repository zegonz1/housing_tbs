import streamlit as st
from model import train_model
import pandas as pd

# === Configuration de la page ===
st.set_page_config(page_title="Estimation de prix immobilier", page_icon="🏡", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🏡 Estimation de prix immobilier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ajustez les caractéristiques de votre bien, puis cliquez sur <b>Valider</b> pour obtenir une estimation.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Chargement du modèle ===
pipeline = train_model()

# === Initialiser l'état ===
if "predict" not in st.session_state:
    st.session_state.predict = False

# === Interface utilisateur ===
st.sidebar.header("🔧 Caractéristiques de la maison")

with st.sidebar.form("formulaire"):
    lot_area = st.number_input("🏡 Surface du terrain (m²)", 0, 100000, 7000, 100)
    bedrooms_above_gr = st.number_input("🛏️ Chambres", 0, 10, 3)
    full_bath = st.number_input("🛁 Salles de bains", 0, 5, 2)
    fireplaces = st.number_input("🔥 Cheminées", 0, 5, 1)
    kitchen_above_gr = st.number_input("🍽️ Cuisines", 0, 5, 1)
    year_built = st.number_input("📅 Année de construction", 1800, 2025, 2005)
    pool_area = st.number_input("🏊 Surface de piscine (m²)", 0, 1000, 0, 10)
    garage_cars = st.number_input("🚗 Garage (voitures)", 0, 10, 2)
    heating = st.selectbox("🔥 Type de chauffage", ["GasA", "GasW", "Grav", "Wall", "Floor", "Steam", "Hot water", "Other"])

    submit = st.form_submit_button("✅ Valider les caractéristiques")

if submit:
    st.session_state.predict = True
    st.session_state.inputs = {
        "LotArea": lot_area,
        "YearBuilt": year_built,
        "Heating": heating,
        "BedroomAbvGr": bedrooms_above_gr,
        "PoolArea": pool_area,
        "GarageCars": garage_cars,
        "Fireplaces": fireplaces,
        "KitchenAbvGr": kitchen_above_gr,
        "FullBath": full_bath
    }

# === Affichage uniquement après clic ===
if st.session_state.predict:
    user_input = pd.DataFrame([st.session_state.inputs])
    predicted_price = pipeline.predict(user_input)[0]

    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #2E86C1;'>💰 Prix estimé de la maison :</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #27AE60;'>{int(predicted_price):,} $</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Estimation basée sur les caractéristiques validées.</p>", unsafe_allow_html=True)
