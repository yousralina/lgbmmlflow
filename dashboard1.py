import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from io import BytesIO

# Configuration
API_URL = "https://mlflowlgbmapi-69a75032435a.herokuapp.com/predict"

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("data_test.csv")
    description = pd.read_csv("HomeCredit_columns_description.csv", index_col=0)
    return data, description

data, description = load_data()

# Sidebar
st.sidebar.title("Prêt à dépenser")
st.sidebar.image("logo.png")
client_id = st.sidebar.selectbox("Sélectionnez un ID client", data['SK_ID_CURR'].unique())
seuil_metier = st.sidebar.slider("Seuil de décision (proba)", 0.0, 1.0, 0.5, 0.01)

# Filtrer le client
client_data = data[data['SK_ID_CURR'] == client_id]

st.title("Dashboard Crédit & Risque Métier")

# Section 1 - Prédiction & Décision
st.header("Décision de Crédit et Évaluation du Risque")

if st.button("Obtenir la décision"):
    client_json = client_data.drop(columns=['SK_ID_CURR']).to_dict(orient="records")[0]
    response = requests.post(API_URL, json=client_json)
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        proba = response.json()["proba"]
        
        st.metric("Probabilité d'accord de crédit", f"{proba:.2f}")

        # Risque métier
        revenu = client_data['AMT_INCOME_TOTAL'].values[0]
        risque_metier = "Faible"
        if revenu < 50000:
            risque_metier = "Élevé"
        elif revenu < 100000:
            risque_metier = "Modéré"

        # Décision
        if proba >= seuil_metier and risque_metier in ["Faible", "Modéré"]:
            decision = "Approbation"
        elif risque_metier == "Élevé":
            decision = "Refus"
        else:
            decision = "Sous conditions"

        st.subheader(f"Décision finale : {decision}")
        st.text(f"Risque métier estimé : {risque_metier}")
        st.text("Motifs :")
        if risque_metier == "Élevé":
            st.text("- Revenu faible")
        else:
            st.text("- Risque métier acceptable")
    else:
        st.error("Erreur lors de la prédiction API.")

# Section 2 - SHAP
st.header("Explication de la Prédiction (SHAP)")
try:
    explainer = shap.TreeExplainer(...)  # Charger votre explainer
    shap_values = explainer.shap_values(client_data.drop(columns=['SK_ID_CURR']))
    fig, ax = plt.subplots()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][0], client_data.drop(columns=['SK_ID_CURR']).iloc[0], show=False)
    st.pyplot(fig)
except Exception as e:
    st.warning("SHAP non disponible ici : assurez-vous d'avoir l'explainer.")

# Section 3 - Comparaison
st.header("Comparaison du Client avec la Population")
st.plotly_chart(px.histogram(data, x="AMT_INCOME_TOTAL", nbins=50, title="Répartition des revenus"))
client_revenu = client_data['AMT_INCOME_TOTAL'].values[0]
st.markdown(f"**Revenu client :** {client_revenu:.2f} €")

# Section 4 - Drift
st.header("Analyse du Drift des Données")
try:
    ref_data = pd.read_csv("data_clean.csv")
    drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    drift_dashboard.calculate(ref_data=ref_data, current_data=data)
    buffer = BytesIO()
    drift_dashboard.save(buffer)
    st.components.v1.html(buffer.getvalue().decode(), height=1000, scrolling=True)
except Exception as e:
    st.warning("Analyse du drift non disponible.")
