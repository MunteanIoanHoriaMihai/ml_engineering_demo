import logging
import pandas as pd
from sklearn.datasets import load_iris
from typing import Tuple

# Configurăm logger-ul doar pentru acest modul
logger = logging.getLogger(__name__)

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Încarcă setul de date Iris și îl returnează separat ca Features (X) și Labels (y).
    """
    try:
        logger.info("Începem încărcarea datelor Iris...")
        
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        
        logger.info(f"Date încărcate cu succes. Dimensiune X: {X.shape}")
        return X, y
        
    except Exception as e:
        logger.error(f"Eroare la încărcarea datelor: {e}")
        raise e

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    O funcție simplă de exemplu pentru curățare.
    Verifică dacă există valori nule.
    """
    if df.isnull().sum().sum() > 0:
        logger.warning("S-au găsit valori lipsă! Se face imputare...")
        return df.fillna(0)
    
    logger.info("Datele sunt curate (fără valori lipsă).")
    return df