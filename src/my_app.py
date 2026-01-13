from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# 1. Definim formatul datelor de intrare (Contractul)
# Pydantic validează automat că primim numere, nu text
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 2. Inițializăm aplicația și logger-ul
app = FastAPI(title="Iris ML API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# 3. Încărcăm modelul O SINGURĂ DATĂ, la pornirea aplicației
# (Nu vrem să-l încărcăm la fiecare cerere, ar fi lent)
MODEL_PATH = "models/iris_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model încărcat cu succes!")
except Exception as e:
    logger.error(f"Nu am putut încărca modelul: {e}")
    model = None

# 4. Definim Endpoint-ul (Ruta)
@app.post("/predict")
def predict_flower(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelul nu este disponibil.")
    
    try:
        # Transformăm input-ul JSON în formatul așteptat de model (DataFrame)
        # Atenție: Ordinea coloanelor trebuie să fie aceeași ca la antrenare!
        input_df = pd.DataFrame([data.dict()])
        
        # === FIX: MAPPING COLOANE ===
        # Modelul a fost antrenat cu nume de coloane specifice (cu spații și cm).
        # Trebuie să redenumim coloanele din API pentru a se potrivi cu cele din model.
        column_mapping = {
            "sepal_length": "sepal length (cm)",
            "sepal_width": "sepal width (cm)",
            "petal_length": "petal length (cm)",
            "petal_width": "petal width (cm)"
        }
        input_df = input_df.rename(columns=column_mapping)

        # Facem predicția
        prediction = model.predict(input_df)
        
        # Returnăm rezultatul
        return {
            "predicted_class": int(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Eroare la predicție: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de verificare (Health Check)
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}