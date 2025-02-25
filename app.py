from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

# Charger le modèle via MLflow
model_uri = "file:///C:/Users/yosra/mlartifacts/970618126747358610/15a09831c7cc44fe906abf30f8b39a22/artifacts/LGBM_Undersampling_Pipeline"
try:
    model = mlflow.pyfunc.load_model(model_uri)

    # Extraire le vrai modèle LightGBM s'il est encapsulé
    if hasattr(model, "unwrap_python_model"):
        model = model.unwrap_python_model().model  # Accède à l'attribut 'model' de CustomModel

    # Vérifier que le modèle possède bien 'predict_proba'
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
    file_path = "C:/Users/yosra/Downloads/Mon_Projet/data_test.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier test introuvable")
    df = pd.read_csv(file_path)
    if "SK_ID_CURR" not in df.columns:
        raise HTTPException(status_code=500, detail="Erreur: colonne SK_ID_CURR absente du fichier CSV")
    df["SK_ID_CURR"] = pd.to_numeric(df["SK_ID_CURR"], errors="coerce")
    df = df.dropna(subset=["SK_ID_CURR"])
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    return df

@app.post("/predict")
def predict(client: ClientID):
    try:
        df = load_data()

        # Vérification si l'ID du client existe dans le dataset
        if client.id_client not in df["SK_ID_CURR"].values:
            raise HTTPException(status_code=404, detail=f"ID client {client.id_client} non trouvé dans le fichier de test.")
        
        # Filtrage des données pour ce client spécifique
        client_data = df[df["SK_ID_CURR"] == client.id_client].drop(columns=["SK_ID_CURR"], errors="ignore")
        
        # Vérification si des données sont disponibles pour ce client
        if client_data.empty:
            raise HTTPException(status_code=400, detail="Données client introuvables après filtrage.")

        # Prédiction de la classe
        prediction = model.predict(client_data)
        
        # Prédiction des probabilités
        prediction_proba = model.predict_proba(client_data)

        # Afficher la probabilité pour le débogage
        print(f"Prediction Proba: {prediction_proba}")

        # Vérification que la probabilité pour la classe 1 est valide
        if prediction_proba.shape[1] < 2:
            raise HTTPException(status_code=500, detail="Le modèle ne renvoie pas de probabilités pour la classe 1.")

        prob = prediction_proba[0][1]

        # Validation que la probabilité est un nombre valide entre 0 et 1
        if prob < 0 or prob > 1:
            raise HTTPException(status_code=500, detail="Probabilité invalide renvoyée.")

        return {"id_client": client.id_client, "prediction": int(prediction[0]), "probability": prob}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
