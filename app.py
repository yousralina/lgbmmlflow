from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import shap
from pydantic import BaseModel
import os

app = FastAPI()

# Définition du chemin du modèle
model_path = os.path.join(os.getcwd(), "model/LGBM_Undersampling_Pipeline")

try:
    model = mlflow.pyfunc.load_model(model_path)

    # Extraire le vrai modèle LightGBM s'il est encapsulé
    if hasattr(model, "unwrap_python_model"):
        model = model.unwrap_python_model().model  

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Le modèle chargé ne possède pas predict_proba()")

    print("✅ Modèle LGBMClassifier chargé avec succès !")

except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle: {str(e)}")

class ClientID(BaseModel):
    id_client: int

@app.get("/")
def home():
    return {"message": "API MLflow en cours d'exécution !"}

def load_data():
    file_path = os.path.join(os.getcwd(), "data_test.csv")  # Chemin relatif

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier test introuvable")

    df = pd.read_csv(file_path)
    if "SK_ID_CURR" not in df.columns:
        raise HTTPException(status_code=500, detail="Erreur: colonne SK_ID_CURR absente du fichier CSV")

    df["SK_ID_CURR"] = pd.to_numeric(df["SK_ID_CURR"], errors="coerce")
    df = df.dropna(subset=["SK_ID_CURR"])
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    return df

# Charger les données et créer l'explainer SHAP
try:
    data_test = load_data()
    feature_columns = [col for col in data_test.columns if col != "SK_ID_CURR"]
    explainer = shap.Explainer(model.predict, data_test[feature_columns])
    print("✅ Explainer SHAP chargé avec succès !")
except Exception as e:
    raise RuntimeError(f"Erreur lors de l'initialisation de SHAP: {str(e)}")

@app.post("/predict")
def predict(client: ClientID):
    try:
        df = load_data()

        if client.id_client not in df["SK_ID_CURR"].values:
            raise HTTPException(status_code=404, detail=f"ID client {client.id_client} non trouvé dans le fichier de test.")
        
        client_data = df[df["SK_ID_CURR"] == client.id_client].drop(columns=["SK_ID_CURR"], errors="ignore")

        if client_data.empty:
            raise HTTPException(status_code=400, detail="Données client introuvables après filtrage.")

        prediction = model.predict(client_data)
        prediction_proba = model.predict_proba(client_data)

        if prediction_proba.shape[1] < 2:
            raise HTTPException(status_code=500, detail="Le modèle ne renvoie pas de probabilités pour la classe 1.")

        prob = prediction_proba[0][1]

        if prob < 0 or prob > 1:
            raise HTTPException(status_code=500, detail="Probabilité invalide renvoyée.")

        # Calcul des valeurs SHAP pour le client
        shap_values = explainer(client_data)
        # Convertir les valeurs SHAP en dictionnaire avec les noms des caractéristiques
        shap_values_dict = {feature_columns[i]: float(shap_values.values[0][i]) for i in range(len(feature_columns))}

        return {
            "id_client": client.id_client,
            "prediction": int(prediction[0]),
            "probability": prob,
            "shap_values": shap_values_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
