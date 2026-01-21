import pandas as pd
import numpy as np
import joblib  # Zum Laden des gespeicherten RF-Modells
import os
from src.paths import MAIN_CSV
from src.paths import VECTORS_CSV
from src.paths import RF_MODEL

def vectorize_rf(main_csv_path, rf_model_path, output_csv_path):
    """
    Generiert Voting-Vektoren für alle Samples in main.csv basierend auf einem gespeicherten Random Forest.
    - main_csv_path: Pfad zur main.csv mit Features + molekuelname
    - rf_model_path: Pfad zum gespeicherten Random Forest (joblib)
    - output_csv_path: Pfad zur CSV, die die Wahrscheinlichkeitsvektoren speichert
    """
    # 1. Random Forest Modell laden
    if not os.path.exists(rf_model_path):
        raise FileNotFoundError(f"Random Forest Model nicht gefunden: {rf_model_path}")
    rf_model = joblib.load(rf_model_path)

    # 2. Main CSV laden
    df = pd.read_csv(main_csv_path).dropna()
    X = df.drop(["molekuelname"], axis=1)
    y = df["molekuelname"]  # optional zum Referenzieren

    # 3. Wahrscheinlichkeitsvektoren berechnen
    prob_vectors = rf_model.predict_proba(X)  # Shape: (n_samples, n_labels)
    label_names = rf_model.classes_          # Name der Labels in der richtigen Reihenfolge

    # 4. In DataFrame packen
    df_probs = pd.DataFrame(prob_vectors, columns=label_names)
    df_probs.insert(0, "molekuelname", y.values)

    df_probs = df_probs.sort_values(by="molekuelname").reset_index(drop=True)


    # 5. Als CSV speichern
    df_probs.to_csv(output_csv_path, index=False)
    print(f"Voting-Vektoren gespeichert in: {output_csv_path}")


# Beispiel-Aufruf
if __name__ == "__main__":
    main_csv = MAIN_CSV
    rf_model_file = RF_MODEL
    output_file = VECTORS_CSV

    vectorize_rf(main_csv, rf_model_file, output_file)
