import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import warnings
import os

st.set_page_config(page_title="Mon Dashboard", page_icon="favicon.ico")

# Ignorer les avertissements
warnings.filterwarnings("ignore")

@st.cache_data
def load_data():
    try:
        # Utilisation des chemins relatifs
        data_test_path = os.path.join(os.path.dirname(__file__), 'data_test.csv')
        data_clean_path = os.path.join(os.path.dirname(__file__), 'data_clean.csv')
        description_path = os.path.join(os.path.dirname(__file__), 'HomeCredit_columns_description.csv')
        
        # Charger les données
        df = pd.read_csv(data_test_path)  # Données de test pour les prédictions
        data_clean = pd.read_csv(data_clean_path)  # Données nettoyées pour l'entraînement du modèle
        description = pd.read_csv(description_path, usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')
        
        return df, data_clean, description
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None, None, None

# Charger les données
df, data_clean, description = load_data()

if df is not None:
    st.write(df.head())  # Afficher les premières lignes des données de test
else:
    st.write("Aucune donnée disponible.")

# Charger le modèle depuis MLflow
@st.cache_resource
def load_model():
    model_uri = "C:/Users/yosra/mlartifacts/970618126747358610/15a09831c7cc44fe906abf30f8b39a22/artifacts/LGBM_Undersampling_Pipeline"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Fonction API prédiction avec Heroku
@st.cache_data
def get_prediction_from_api(client_id):
    API_url = f"https://apimlflowlgbm-932ffe55319a-6cd14c48bc7e.herokuapp.com/"  # URL de votre API Heroku
    data = json.dumps({"id_client": client_id})
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API: {e}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #7451EB;'>💳 Prêt à dépenser</h1>", unsafe_allow_html=True)
    
    # Choisir l'ID client
    st.write("## ID Client 🆔")
    id_list = df["SK_ID_CURR"].values
    id_client = st.selectbox("Sélectionner l'identifiant du client", id_list)
    
    # Options pour le dashboard
    st.write("## Actions à effectuer ⚙️")
    show_credit_decision = st.checkbox("Afficher la décision de crédit 📊")
    show_client_details = st.checkbox("Afficher les informations du client 👤")
    show_client_comparison = st.checkbox("Comparer aux autres clients 📈")
    show_data_drift = st.checkbox("Analyse du Data Drift 📉")
    shap_general = st.checkbox("Explication des décisions SHAP 🔍")
    show_feature_description = st.checkbox("Afficher la description des features 📖")

# Page d'accueil
st.markdown("<h1 style='text-align: center; color: #7451EB;'>Dashboard de Scoring Crédit 🏦</h1>", unsafe_allow_html=True)

# Section : À quoi sert ce dashboard ?
st.markdown("""
<div style='background-color: #1C1C1C; padding: 20px; border-radius: 10px;'>
    <h2 style='color: #7451EB;'>📌 À quoi sert ce dashboard ?</h2>
    <p style='color: #FFFFFF;'>
        Ce dashboard est conçu pour aider les agents de crédit à prendre des décisions éclairées en fournissant :
        <ul style='color: #FFFFFF;'>
            <li><strong>📊 Une prédiction de scoring</strong> : Estime la probabilité de défaut de paiement d'un client.</li>
            <li><strong>🔍 Des explications SHAP</strong> : Explique les facteurs influençant la décision du modèle.</li>
            <li><strong>📉 Une analyse du Data Drift</strong> : Détecte les changements dans les données actuelles par rapport aux données d'entraînement.</li>
            <li><strong>📋 Des informations détaillées</strong> : Affiche les informations du client et les compare à la moyenne des autres clients.</li>
        </ul>
    </p>
</div>
""", unsafe_allow_html=True)

# Vérification ID Client
if int(id_client) in df["SK_ID_CURR"].values:
    client_info = df[df['SK_ID_CURR'] == int(id_client)]

    if not client_info.empty:
        st.success(f"Données trouvées pour l'ID client {id_client}.")
        
        # 📋 **Informations du client**
        if show_client_details:
            st.header("📋 Informations du client sélectionné")
            
            info_cols = {
                'DAYS_BIRTH': "Âge 🎂",
                'AMT_INCOME_TOTAL': "Revenus 💰",
                'AMT_CREDIT': "Montant du Crédit 💸",
                'AMT_ANNUITY': "Montant Annuités 📅",
                'NAME_EDUCATION_TYPE': "Niveau d'Éducation 🎓",
                'OCCUPATION_TYPE': "Emploi 💼"
            }

            # Sélection des colonnes disponibles
            available_cols = [col for col in info_cols.keys() if col in client_info.columns]
            client_data = client_info[available_cols].rename(columns=info_cols)
            
            # Conversion de l'âge en années
            if "Âge 🎂" in client_data.columns:
                client_data["Âge 🎂"] = abs(client_data["Âge 🎂"] // 365)
            
            st.table(client_data.T)

        # 📊 **Comparaison du client avec les autres (Graphique en anneau)**
        if show_client_comparison:
            st.header("📊 Comparaison avec les autres clients")
            
            # Données pour le graphique en anneau
            labels = ["Revenus", "Montant Crédit", "Montant Annuités"]
            client_values = [client_data["Revenus 💰"].values[0], client_data["Montant du Crédit 💸"].values[0], client_data["Montant Annuités 📅"].values[0]]
            avg_values = [data_clean["AMT_INCOME_TOTAL"].mean(), data_clean["AMT_CREDIT"].mean(), data_clean["AMT_ANNUITY"].mean()]

            # Création du graphique en anneau
            fig = go.Figure()

            # Ajouter les valeurs du client
            fig.add_trace(go.Pie(
                labels=labels,
                values=client_values,
                hole=0.5,  # Taille du trou au centre (pour créer un anneau)
                name="Client Sélectionné",
                marker_colors=["#7451EB", "#A78BFA", "#D1C4E9"],  # Couleurs OpenClassrooms
                textinfo="percent+label",
                domain={"x": [0, 0.45]}  # Position du premier anneau
            ))

            # Ajouter les valeurs moyennes des autres clients
            fig.add_trace(go.Pie(
                labels=labels,
                values=avg_values,
                hole=0.5,
                name="Moyenne des Clients",
                marker_colors=["#7451EB", "#A78BFA", "#D1C4E9"],  
                textinfo="percent+label",
                domain={"x": [0.55, 1]}  # Position du deuxième anneau
            ))

            # Mise en forme du graphique
            fig.update_layout(
                title="Comparaison Client vs Moyenne (Graphique en Anneau)",
                annotations=[
                    {"text": "Client", "x": 0.2, "y": 0.5, "font_size": 14, "showarrow": False, "font_color": "#FFFFFF"},
                    {"text": "Moyenne", "x": 0.8, "y": 0.5, "font_size": 14, "showarrow": False, "font_color": "#FFFFFF"}
                ],
                paper_bgcolor="#1C1C1C",  # Fond sombre
                plot_bgcolor="#1C1C1C",   # Fond sombre
                font_color="#FFFFFF"      # Texte blanc
            )

            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig)

# 📉 **Analyse du Data Drift**
if show_data_drift:
    st.header("📉 Analyse du Data Drift")
    
    # Nettoyage des noms de colonnes
    data_clean.columns = data_clean.columns.str.strip().str.lower()
    df.columns = df.columns.str.strip().str.lower()
    
    # Générer un rapport d'Evidently pour analyser la dérive des données
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=data_clean, current_data=df)
    
    # Afficher le rapport dans Streamlit
    st.write(report.show_html())

# 🔍 **Explications SHAP**
if shap_general:
    st.header("🔍 Explications des décisions avec SHAP")

    model = load_model()
    if model:
        # Charger les explainer SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df.drop(columns=['SK_ID_CURR']))
        shap.initjs()

        # Affichage du summary plot
        st.shap.summary_plot(shap_values, df.drop(columns=['SK_ID_CURR']))
