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
import os

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

# Fonction API prédiction avec Heroku
@st.cache_data
def get_prediction_from_api(client_id):
    API_url = f"https://applicationmlflowdash-6ac6d5ea24de.herokuapp.com/predict"  # URL de votre API Heroku
    data = json.dumps({"id_client": client_id})
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API: {e}")
        return None


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




import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 📉 **Analyse du Data Drift**
if show_data_drift:
    st.header("📉 Analyse du Data Drift")

    try:
        # Nettoyage des noms de colonnes
        data_clean.columns = data_clean.columns.str.strip().str.lower()
        df.columns = df.columns.str.strip().str.lower()

        # Trouver les colonnes communes
        common_columns = data_clean.columns.intersection(df.columns)

        # Filtrer les datasets pour ne garder que les colonnes communes
        reference_data = data_clean[common_columns]
        current_data = df[common_columns]

        if set(reference_data.columns) == set(current_data.columns):
            # Générer le rapport de Data Drift
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=reference_data, current_data=current_data)

            # Extraire les résultats sous forme de dictionnaire
            drift_results = drift_report.as_dict()

            # 🔍 Vérification de la structure des résultats
            if "metrics" in drift_results and len(drift_results["metrics"]) > 0:
                drift_metrics = drift_results["metrics"][0]["result"]
                
                if "drift_by_columns" in drift_metrics:
                    drift_data = []
                    for col_name, col_metrics in drift_metrics["drift_by_columns"].items():
                        drift_score = col_metrics["drift_score"]
                        p_value = col_metrics["p_value"]
                        threshold = drift_metrics["threshold"]
                        drift_status = "Drift" if drift_score > threshold else "Stable"
                        drift_data.append([col_name, drift_score, p_value, drift_status])

                    drift_df = pd.DataFrame(drift_data, columns=["Feature", "Drift Score", "p-Value", "Status"])

                    # 📊 **Graphique en barre du drift**
                    st.write("## 📈 Niveau de Drift par Colonne")
                    fig_drift = px.bar(drift_df, x="Feature", y="Drift Score", color="Status",
                                       color_discrete_map={"Drift": "red", "Stable": "green"},
                                       title="Score de Drift par Variable",
                                       labels={"Drift Score": "Score de dérive"})
                    fig_drift.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_drift, use_container_width=True)

                    # 🔥 **Heatmap des p-values du Drift**
                    st.write("## 🔥 Heatmap des p-values")
                    heatmap_data = pd.DataFrame({"Feature": drift_df["Feature"], "p-Value": drift_df["p-Value"]})
                    fig_heatmap = px.imshow([heatmap_data["p-Value"]], 
                                            labels=dict(x="Feature", y="p-Value"),
                                            x=heatmap_data["Feature"], 
                                            y=["p-Value"],
                                            color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    # 📊 **Comparaison des distributions**
                    st.write("## 📊 Comparaison des distributions")
                    selected_feature = st.selectbox("Choisir une variable à comparer", drift_df["Feature"])
                    fig_dist = px.histogram(pd.concat([reference_data[selected_feature].rename("Référence"), 
                                                       current_data[selected_feature].rename("Actuel")], axis=1),
                                            barmode="overlay", 
                                            title=f"Distribution de {selected_feature} (Référence vs Actuel)")
                    fig_dist.update_traces(opacity=0.6)
                    st.plotly_chart(fig_dist, use_container_width=True)

                else:
                    st.error("⚠️ Aucun drift détecté ou problème d'extraction des données.")
            else:
                st.error("🚨 Problème lors de la génération du rapport Evidently.")

        else:
            st.error("⚠️ Les colonnes de référence et actuelles ne correspondent pas. Vérifiez les données.")

    except Exception as e:
        st.error(f"🚨 Erreur lors de l'analyse du Data Drift : {str(e)}")
        
import streamlit as st
import plotly.graph_objects as go

# 🔢 **Affichage de la décision de crédit**
if show_credit_decision:
    st.header('📊 Scoring et décision du modèle')

    # Seuil personnalisable pour la décision de crédit
    threshold_credit = st.slider("Définir le seuil de refus de crédit (%)", min_value=0, max_value=100, value=50, step=1) / 100

    with st.spinner('🔄 Chargement du score du client...'):
        prediction_data = get_prediction_from_api(id_client)

        if prediction_data:
            classe_predite = prediction_data['prediction']
            proba = prediction_data.get('probability', None)

            if proba is None or not (0 <= proba <= 1):
                st.error("Erreur: La probabilité retournée par l'API est invalide.")
            else:
                decision = '🚫 Mauvais prospect (Crédit Refusé)' if proba >= threshold_credit else '✅ Bon prospect (Crédit Accordé)'
                client_score = round(proba * 100, 2)

                # Affichage
                left_column, right_column = st.columns((1, 2))

                left_column.write(f'**Risque de défaut : {client_score}%**')
                left_column.write(f'**Décision :** :{ "red_circle" if proba >= threshold_credit else "green_circle"}: **{decision}**')

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
