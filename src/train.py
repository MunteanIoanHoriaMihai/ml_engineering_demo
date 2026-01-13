import logging
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os           # <--- NOU: Pentru a crea foldere
import joblib       # <--- NOU: Pentru a salva modelul

# Importăm funcțiile noastre din celălalt fișier
from data_loader import load_data, clean_data

# Configurare Logging Globală (pentru a vedea mesajele în consolă)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=== Start Pipeline Antrenare ===")

    # 1. Încărcare Date
    X, y = load_data()

    # 2. Curățare (opțional, doar ca exemplu)
    X = clean_data(X)

    # 3. Split Train/Test
    logger.info("Împărțim datele în Train și Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Initializare Model
    # Aici setăm hiperparametrii explicit, nu lăsăm default
    n_estimators = 100
    max_depth = 5
    logger.info(f"Inițializăm RandomForest (n_estimators={n_estimators}, max_depth={max_depth})")
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # 5. Antrenare
    logger.info("Antrenăm modelul...")
    model.fit(X_train, y_train)

    # 6. Evaluare
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    logger.info(f"=== Antrenare Completă. Acuratețe pe Test: {acc:.4f} ===")


    # SALVAREA MODELULUI

    # Definim calea unde vrem să salvăm
    save_dir = "models"
    model_filename = "iris_model.joblib"
    save_path = os.path.join(save_dir, model_filename)

    # Ne asigurăm că folderul există (dacă nu, îl creăm)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"Creat folderul: {save_dir}")

    # Salvăm întregul obiect 'clf' (clasa noastră)
    # Asta va salva și modelul sklearn din interiorul ei, și orice altă stare
    joblib.dump(model, save_path)
    
    logger.info(f"✅ Model salvat cu succes în: {save_path}")
    logger.info("=== Pipeline Finalizat ===")

if __name__ == "__main__":
    main()