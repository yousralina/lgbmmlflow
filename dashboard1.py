import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")

@st.cache_data
def load_data():
    try:
        # Utilisation des chemins relatifs
        data_test_path = os.path.join(os.path.dirname(__file__),  'data_test.csv')
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

# Fonction API prédiction
@st.cache_data
def get_prediction_from_api(client_id):
    API_url = f"https://applicationmlflowdash-6ac6d5ea24de.herokuapp.com/predict"
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

import plotly.graph_objects as go
import streamlit as st

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

        # 📊 **Comparaison du client avec les autres (Graphique en anneau et barre)**
        if show_client_comparison:
            st.header("📊 Comparaison avec les autres clients")
            
            # Données pour les graphiques
            labels = ["Revenus", "Montant Crédit", "Montant Annuités"]
            client_values = [client_data["Revenus 💰"].values[0], client_data["Montant du Crédit 💸"].values[0], client_data["Montant Annuités 📅"].values[0]]
            avg_values = [data_clean["AMT_INCOME_TOTAL"].mean(), data_clean["AMT_CREDIT"].mean(), data_clean["AMT_ANNUITY"].mean()]

            # Création du graphique en anneau
            fig = go.Figure()

            # Ajouter les valeurs du client dans le premier anneau
            fig.add_trace(go.Pie(
                labels=labels,
                values=client_values,
                hole=0.5,  # Taille du trou au centre (pour créer un anneau)
                name="Client Sélectionné",
                marker_colors=["#1F77B4", "#FF7F0E", "#2CA02C"],  
                textinfo="percent+label",
                domain={"x": [0, 0.45]}  # Position du premier anneau
            ))

            # Ajouter les valeurs moyennes des autres clients dans le deuxième anneau
            fig.add_trace(go.Pie(
                labels=labels,
                values=avg_values,
                hole=0.5,
                name="Moyenne des Clients",
                marker_colors=["#1F77B4", "#FF7F0E", "#2CA02C"],  # Couleurs adaptées
                textinfo="percent+label",
                domain={"x": [0.55, 1]}  # Position du deuxième anneau
            ))

            # Mise en forme du graphique en anneau
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

            # Afficher le graphique en anneau dans Streamlit
            st.plotly_chart(fig)

            # 📊 **Graphique en barre pour comparaison des montants**
            st.write("### Graphique en Barre des Montants (Client vs Moyenne)")

            bar_fig = go.Figure()

            bar_fig.add_trace(go.Bar(
                x=labels,
                y=client_values,
                name="Client Sélectionné",
                marker_color="#1F77B4",  # Couleur bleu
                text=client_values,
                textposition='auto',
                hoverinfo='x+y'
            ))

            bar_fig.add_trace(go.Bar(
                x=labels,
                y=avg_values,
                name="Moyenne des Clients",
                marker_color="#FF7F0E",  # Couleur orange
                text=avg_values,
                textposition='auto',
                hoverinfo='x+y'
            ))

            # Mise en forme du graphique en barre
            bar_fig.update_layout(
                title="Comparaison des Montants (Client vs Moyenne)",
                barmode="group",  # Mode groupe pour les barres côte à côte
                paper_bgcolor="#1C1C1C",  # Fond sombre
                plot_bgcolor="#1C1C1C",   # Fond sombre
                font_color="#FFFFFF",     # Texte blanc
                xaxis_title="Critères",
                yaxis_title="Montants"
            )

            # Afficher le graphique en barre dans Streamlit
            st.plotly_chart(bar_fig)

# Fonction pour charger le modèle LGBM à partir du pipeline
@st.cache_resource
def load_model():
    model_uri = r"C:\Users\yosra\mlartifacts\970618126747358610\15a09831c7cc44fe906abf30f8b39a22\artifacts\mon_projet_api\model\LGBM_Undersampling_Pipeline"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Fonction pour obtenir la prédiction à partir de l'API
@st.cache_data
def get_prediction_from_api(client_id):
    API_url = f"https://applicationmlflowdash-6ac6d5ea24de.herokuapp.com/predict"
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
    model_uri = r"C:\Users\yosra\mlartifacts\970618126747358610\15a09831c7cc44fe906abf30f8b39a22\artifacts\mon_projet_api\model\LGBM_Undersampling_Pipeline"
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
    model_uri = r"C:\Users\yosra\mlartifacts\970618126747358610\15a09831c7cc44fe906abf30f8b39a22\artifacts\mon_projet_api\model\LGBM_Undersampling_Pipeline"
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
    model_uri = r"C:\Users\yosra\mlartifacts\970618126747358610\15a09831c7cc44fe906abf30f8b39a22\artifacts\mon_projet_api\model\LGBM_Undersampling_Pipeline"
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
    model_uri = r"C:\Users\yosra\mlartifacts\970618126747358610\15a09831c7cc44fe906abf30f8b39a22\artifacts\mon_projet_api\model\LGBM_Undersampling_Pipeline"
    try:
        # Charger le modèle LightGBM directement depuis le chemin local
        model = mlflow.lightgbm.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import streamlit as st

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

                # Appliquer les couleurs sombres
                ax.set_facecolor("#1C1C1C")  # Fond sombre
                fig.patch.set_facecolor("#1C1C1C")  # Fond sombre
                plt.setp(ax.get_xticklabels(), color='#FFFFFF')  # Texte des labels en blanc
                plt.setp(ax.get_yticklabels(), color='#FFFFFF')  # Texte des labels en blanc
                ax.spines['top'].set_color('#FFFFFF')  # Bordure en blanc
                ax.spines['right'].set_color('#FFFFFF')  # Bordure en blanc
                ax.spines['bottom'].set_color('#FFFFFF')  # Bordure en blanc
                ax.spines['left'].set_color('#FFFFFF')  # Bordure en blanc

                st.pyplot(fig)

            else:
                st.error("Le modèle chargé n'est pas un modèle LightGBM.")
        except Exception as e:
            st.error(f"Erreur lors du calcul de SHAP : {e}")




# 📉 **Analyse du Data Drift**
if show_data_drift:
    st.header("📉 Analyse du Data Drift")


    # Nettoyage et alignement des colonnes
    data_clean.columns = data_clean.columns.str.strip().str.lower()
    df.columns = df.columns.str.strip().str.lower()
    
    # Sélection des colonnes communes
    common_columns = list(set(data_clean.columns).intersection(set(df.columns)))
    reference_data = data_clean[common_columns]
    current_data = df[common_columns]

    if not common_columns:
        st.error("Aucune colonne commune trouvée entre les datasets !")
    else:
        # Générer le rapport de Data Drift avec Evidently
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_data, current_data=current_data)

        # Extraire les résultats du rapport
        drift_results = drift_report.as_dict()

        # Afficher la structure du rapport pour vérifier où se trouvent les données
        st.write(drift_results)  # Afficher le rapport complet pour analyser la structure

        # Vérifier si la clé 'drift_by_columns' existe dans les résultats
        if 'metrics' in drift_results and len(drift_results['metrics']) > 0:
            drift_features = drift_results['metrics'][0].get('result', {}).get('drift_by_columns', {})
            
            if drift_features:
                # Transformer les résultats en DataFrame
                drift_df = pd.DataFrame(drift_features).T
                drift_df = drift_df.reset_index().rename(columns={'index': 'Feature'})
                
                # Filtrer les features présentant un drift
                drifted_features = drift_df[drift_df['drift_detected'] == True]

                # 🎨 **Visualisation du Data Drift avec Plotly**
                st.write("### Visualisation du Data Drift")
                
                # Sélection des features les plus impactées
                top_drifted_features = drifted_features.sort_values(by='drift_score', ascending=False).head(5)

                if not top_drifted_features.empty:
                    fig = go.Figure()
                    for feature in top_drifted_features['Feature']:
                        fig.add_trace(go.Box(
                            y=reference_data[feature],
                            name=f"{feature} (Référence)",
                            marker_color='blue'
                        ))
                        fig.add_trace(go.Box(
                            y=current_data[feature],
                            name=f"{feature} (Actuel)",
                            marker_color='red'
                        ))

                    fig.update_layout(
                        title="Comparaison des distributions des features avec Data Drift",
                        xaxis_title="Features",
                        yaxis_title="Valeurs",
                        boxmode='group',
                        paper_bgcolor="#1C1C1C",  # Fond sombre
                        plot_bgcolor="#1C1C1C",
                        font_color="#FFFFFF"
                    )
                    st.plotly_chart(fig)
                else:
                    st.info("Aucune variable avec un drift significatif à afficher.")
            else:
                st.warning("Aucune dérive détectée dans les colonnes.")
        else:
            st.error("Le rapport de dérive ne contient pas de données valides.")

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
 # Seuil personnalisable pour la décision de crédit
threshold_credit = st.slider("Définir le seuil de refus de crédit (%)", min_value=0, max_value=100, value=50, step=1) / 100

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

 # 🎯 **Graphique circulaire moderne (Gauge Chart)**
fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=client_score,
                    title={"text": "Risque de défaut (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "white"},
                        "steps": [
                            {"range": [0, 40], "color": "green"},
                            {"range": [40, 70], "color": "orange"},
                            {"range": [70, 100], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": threshold_credit * 100,
                        },
                    }
                ))

right_column.plotly_chart(fig_gauge)                
   


        # 📖 **Description des features**
if show_feature_description:
            st.header("📖 Description des features")
            feature_to_describe = st.selectbox("Sélectionner une feature", description.index)
            st.write(f"**{feature_to_describe}** : {description.loc[feature_to_describe, 'Description']}")

else:
    st.error(f"L'ID client {id_client} n'existe pas dans la base de données.")
