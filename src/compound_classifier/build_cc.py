import joblib
import os
from src.compound_classifier import CompoundClassifier
from src.paths import MODELS_DIR

# Pfade
rf_path = os.path.join(MODELS_DIR, "rf_model.joblib")
dbscan_path = os.path.join(MODELS_DIR, "dbscan_classifier.pkl")
compound_path = os.path.join(MODELS_DIR, "compound_classifier.pkl")

def build_compound_classifier():
    print("Lade Random Forest...")
    print(f" -> {rf_path}")

    print("Lade DBSCAN...")
    print(f" -> {dbscan_path}")

    # CompoundClassifier initialisieren
    clf = CompoundClassifier(rf_path, dbscan_path)

    # Speichern
    print("\nSpeichere Compound Classifier...")
    joblib.dump(clf, compound_path)

    print(f"FERTIG! Compound Classifier gespeichert unter:\n  {compound_path}")


if __name__ == "__main__":
    build_compound_classifier()
