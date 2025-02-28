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
import streamlit as st
st.set_page_config(page_title="Mon Dashboard", page_icon="favicon.ico")


# Ignorer les avertissements
warnings.filterwarnings("ignore")

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_test.csv')  # Données de test pour les prédictions
        data_clean = pd.read_csv('data_clean.csv')  # Données nettoyées pour l'entraînement du modèle
        description = pd.read_csv('HomeCredit_columns_description.csv', usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')
        return df, data_clean, description
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None, None, None

df, data_clean, description = load_data()

# Charger le modèle depuis MLflow
@st.cache_resource
def load_model():
    # Option 1 : Charger depuis MLflow
    # model_uri = "runs:/15a09831c7cc44fe906abf30f8b39a22/LGBM_Undersampling_Pipeline"
    
    # Option 2 : Charger directement depuis le fichier
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
    API_url = f"https://mlflowlgbmapi-69a75032435a.herokuapp.com/predict"  # URL de votre API Heroku
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
                marker_colors=["#7451EB", "#A78BFA", "#D1C4E9"],  # Couleurs OpenClassrooms
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

# Fonction pour charger le modèle LGBM à partir du pipeline
@st.cache_resource
def load_model():
    model_uri = "C:/Users/yosra/mlartifacts/970618126747358610/15a09831c7cc44fe906abf30f8b39a22/artifacts/LGBM_Undersampling_Pipeline"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Fonction pour obtenir la prédiction à partir de l'API
@st.cache_data
def get_prediction_from_api(client_id):
    API_url = f"https://mlflowlgbmapi-69a75032435a.herokuapp.com/predict"
    data = json.dumps({"id_client": client_id})
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API: {e}")
        return None

# Section pour l'explication avec SHAP
import mlflow
import shap
import lightgbm as lgb

# Fonction pour charger le modèle pyfunc depuis MLflow
@st.cache_resource
def load_model():
    model_uri = "C:/Users/yosra/mlartifacts/970618126747358610/15a09831c7cc44fe906abf30f8b39a22/artifacts/LGBM_Undersampling_Pipeline"
    try:
        # Charger le modèle en tant que pyfunc
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Section pour l'explication avec SHAP
import mlflow
import shap
import lightgbm as lgb

# Fonction pour charger le modèle pyfunc depuis MLflow
@st.cache_resource
def load_model():
    model_uri = "C:/Users/yosra/mlartifacts/970618126747358610/15a09831c7cc44fe906abf30f8b39a22/artifacts/LGBM_Undersampling_Pipeline"
    try:
        # Charger le modèle en tant que pyfunc
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Section pour l'explication avec SHAP
import mlflow
import shap
import lightgbm as lgb
import streamlit as st
import matplotlib.pyplot as plt

# Fonction pour charger le modèle LightGBM depuis MLflow
@st.cache_resource
def load_model():
    model_uri = "mlflow-artifacts:/970618126747358610/9367d103f9b14eafbfad7071648c2164/artifacts/LGBM_Undersampling_Pipeline"
    try:
        # Charger le modèle LightGBM directement
        model = mlflow.lightgbm.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Section pour l'explication avec SHAP
import mlflow
import shap
import lightgbm as lgb
import streamlit as st
import matplotlib.pyplot as plt

# Fonction pour charger le modèle LightGBM depuis un chemin local
@st.cache_resource
def load_model():
    model_uri = "file:///C:/Users/yosra/mlartifacts/970618126747358610/9367d103f9b14eafbfad7071648c2164/artifacts/LGBM_Undersampling_Pipeline"
    try:
        # Charger le modèle LightGBM directement depuis le chemin local
        model = mlflow.lightgbm.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Section pour l'explication avec SHAP
if shap_general:
    st.header("🔍 Explication des décisions avec SHAP")
    model = load_model()

    if model is not None:
        try:
            # Vérifier que le modèle est de type LightGBM
            if isinstance(model, lgb.sklearn.LGBMClassifier):
                # Préparer les données sans la colonne cible
                client_data_without_target = client_info.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")

                # Calculer les valeurs SHAP
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(client_data_without_target)

                # Afficher le graphique SHAP
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, client_data_without_target, plot_type="bar", show=False)
                ax.set_facecolor("#1C1C1C")  # Fond sombre
                fig.patch.set_facecolor("#1C1C1C")  # Fond sombre
                st.pyplot(fig)
            else:
                st.error("Le modèle chargé n'est pas un modèle LightGBM.")
        except Exception as e:
            st.error(f"Erreur lors du calcul de SHAP : {e}")



# 📉 **Analyse du Data Drift**
if show_data_drift:
    st.header("📉 Analyse du Data Drift")
    
    # Nettoyage des noms de colonnes
    data_clean.columns = data_clean.columns.str.strip().str.lower()
    df.columns = df.columns.str.strip().str.lower()
    
    # Trouver les colonnes communes
    common_columns = list(set(data_clean.columns).intersection(set(df.columns)))
    
    # Filtrer les datasets pour ne garder que les colonnes communes
    reference_data = data_clean[common_columns]
    current_data = df[common_columns]
    
    # Vérifier si les colonnes sont bien identiques
    if set(reference_data.columns) == set(current_data.columns):
        # Générer le rapport de Data Drift
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_data, current_data=current_data)
        
        # Extraire les résultats sous forme de dictionnaire
        drift_results = drift_report.as_dict()
        
        # 🔹 **Afficher le tableau des résultats**
        st.write("## 📊 Résumé des métriques du Data Drift")
        drift_data = []
        for feature in drift_results["metrics"][0]["result"]["drift_by_columns"]:
            col_name = feature
            drift_score = drift_results["metrics"][0]["result"]["drift_by_columns"][col_name]["drift_score"]
            p_value = drift_results["metrics"][0]["result"]["drift_by_columns"][col_name]["p_value"]
            threshold = drift_results["metrics"][0]["result"]["threshold"]
            drift_detected = "Drift" if drift_score > threshold else "Stable"
            
            drift_data.append([col_name, drift_score, p_value, threshold, drift_detected])
        
        drift_df = pd.DataFrame(drift_data, columns=["Feature", "Drift Score", "p-Value", "Threshold", "Status"])
        st.dataframe(drift_df.style.applymap(lambda x: "background-color: #FFDDC1" if x == "Drift" else "background-color: #C1FFD7", subset=["Status"]))
        
        # 📈 **Graphique interactif de la dérive des variables**
        st.write("## 📈 Visualisation de la dérive des colonnes")
        fig = px.bar(drift_df, x="Feature", y="Drift Score", color="Status",
                     color_discrete_map={"Drift": "red", "Stable": "green"},
                     title="Niveau de Drift par Colonne",
                     labels={"Drift Score": "Score de dérive"})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # 🔎 **Afficher le rapport HTML interactif**
        st.write("## 📄 Rapport détaillé du Data Drift")
        st.components.v1.html(drift_report.get_html(), height=600, scrolling=True)
    
    else:
        st.error("⚠️ Les colonnes de référence et actuelles ne correspondent pas. Vérifiez les données.")
        st.write("Colonnes de référence (data_clean):", reference_data.columns.tolist())
        st.write("Colonnes actuelles (df):", current_data.columns.tolist())

# Affichage de la décision de crédit
if show_credit_decision:
    st.header('📊 Scoring et décision du modèle')
    with st.spinner('🔄 Chargement du score du client...'):
        prediction_data = get_prediction_from_api(id_client)
        
        if prediction_data:
            classe_predite = prediction_data['prediction']
            proba = prediction_data.get('probability', None)
            
            # Vérification de la validité de la probabilité
            if proba is None or not (0 <= proba <= 1):
                st.error("Erreur: La probabilité retournée par l'API est invalide.")
            else:
                decision = '🚫 Mauvais prospect (Crédit Refusé)' if classe_predite == 1 else '✅ Bon prospect (Crédit Accordé)'
                client_score = round(proba * 100, 2)
                
                # Affichage de la comparaison des risques
                left_column, right_column = st.columns((1, 2))
                
                left_column.markdown(f'Risque de défaut: **{client_score}%**')
                left_column.markdown(f'Décision: <span style="color:{"red" if classe_predite == 1 else "green"}">**{decision}**</span>', unsafe_allow_html=True)
                
                # Graphique interactif Plotly
                fig = go.Figure(go.Bar(
                    x=["Risque de défaut", "Bon prospect"],
                    y=[client_score, 100 - client_score],
                    marker_color=["red", "green"]
                ))
                fig.update_layout(
                    title="Comparaison Risque de défaut vs Bon Prospect",
                    xaxis_title="Scénarios",
                    yaxis_title="Probabilité (%)"
                )
                right_column.plotly_chart(fig)
   


        # 📖 **Description des features**
        if show_feature_description:
            st.header("📖 Description des features")
            feature_to_describe = st.selectbox("Sélectionner une feature", description.index)
            st.write(f"**{feature_to_describe}** : {description.loc[feature_to_describe, 'Description']}")

else:
    st.error(f"L'ID client {id_client} n'existe pas dans la base de données.")

