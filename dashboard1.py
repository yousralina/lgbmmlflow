import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import warnings

st.set_page_config(page_title="Mon Dashboard", page_icon="favicon.ico")

warnings.filterwarnings("ignore")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_test.csv')
        data_clean = pd.read_csv('data_clean.csv')
        description = pd.read_csv('HomeCredit_columns_description.csv', usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')
        return df, data_clean, description
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None, None, None

df, data_clean, description = load_data()

@st.cache_resource
def load_model():
    model_uri = "mlflow-artifacts:/970618126747358610/9367d103f9b14eafbfad7071648c2164/artifacts/LGBM_Undersampling_Pipeline"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

@st.cache_data
def get_prediction_from_api(client_id):
    API_url = "https://apimlflowlgbm-932ffe55319a.herokuapp.com/predict"
    data = json.dumps({"id_client": client_id})
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API: {e}")
        return None

st.header("🔍 Explication des décisions avec SHAP")
model = load_model()

if model is not None:
    try:
        client_data = data_clean.loc[data_clean["SK_ID_CURR"] == df["SK_ID_CURR"].iloc[0]]
        if not client_data.empty:
            client_data_without_target = client_data.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
            explainer = shap.Explainer(model.predict, client_data_without_target)
            shap_values = explainer(client_data_without_target)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, client_data_without_target, plot_type="bar", show=False)
            st.pyplot(fig)
        else:
            st.error("Aucune donnée client trouvée.")
    except Exception as e:
        st.error(f"Erreur lors du calcul de SHAP : {e}")

    # Ajout de nouveaux graphiques de comparaison client
    st.header("📊 Comparaison du client avec les autres")
    
    if "TARGET" in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df["TARGET"], kde=True, bins=30, color="blue", label="Autres clients", ax=ax)
        ax.axvline(client_data["TARGET"].values[0], color="red", linestyle="--", label="Client sélectionné")
        ax.set_title("Distribution des scores de crédit")
        ax.legend()
        st.pyplot(fig)
    
        fig, ax = plt.subplots()
        sns.violinplot(x=df["TARGET"], color="lightblue", ax=ax)
        ax.axhline(client_data["TARGET"].values[0], color="red", linestyle="--", label="Client sélectionné")
        ax.set_title("Distribution des scores en violon")
        st.pyplot(fig)
    
        fig, ax = plt.subplots()
        sns.boxplot(y=df["TARGET"], color="orange", ax=ax)
        ax.axhline(client_data["TARGET"].values[0], color="red", linestyle="--", label="Client sélectionné")
        ax.set_title("Boxplot des scores de crédit")
        st.pyplot(fig)
