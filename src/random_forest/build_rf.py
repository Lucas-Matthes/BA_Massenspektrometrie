# src/random_forest/build_rf.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from src.paths import MAIN_CSV
from src.paths import RF_MODEL

# Baut ein RandomForest Modell basierend auf main.csv und speichert es für spätere Nutzung.

def build_rf_model(input_csv, output_model_path, n_estimators=50, random_state=42):

    # CSV laden
    df = pd.read_csv(input_csv).dropna()
    X = df.drop(["molekuelname"], axis=1)
    y = df["molekuelname"]

    # Random Forest trainieren
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X, y)

    # Model speichern
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(rf_model, output_model_path)
    print(f"Random Forest Modell gespeichert unter: {output_model_path}")

    return rf_model

if __name__ == "__main__":
    # Beispiel-Pfade
    input_csv = MAIN_CSV
    output_model_path = RF_MODEL
    build_rf_model(input_csv, output_model_path, n_estimators=50000)
