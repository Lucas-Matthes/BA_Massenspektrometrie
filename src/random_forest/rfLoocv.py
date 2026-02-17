import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from src.paths import MAIN_CSV

def run_loocv(input_csv, n_estimators=50):
    """
    Führt LOOCV für den Random Forest auf der Main-Daten CSV durch.
    Gibt Accuracy und Standard Error (SEM) zurück.
    """
    df = pd.read_csv(input_csv).dropna()
    X = df.drop(["molekuelname"], axis=1)
    y = df["molekuelname"]

    loo = LeaveOneOut()
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf_model.fit(X_train, y_train)
        pred = rf_model.predict(X_test)

        y_true.append(y_test.values[0])
        y_pred.append(pred[0])

    accuracies = (np.array(y_true) == np.array(y_pred)).astype(int)
    mean_acc = np.mean(accuracies)
    sem_acc = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))

     # Konfusionsmatrix erstellen
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Konfusionsmatrix")
    plt.tight_layout()
    plt.show()

    print(f"LOOCV Ergebnis auf Main-Daten:")
    print(f"n_estimators = {n_estimators}")
    print(f"Accuracy = {mean_acc:.3f} ± {sem_acc:.3f} (SEM)")


    return mean_acc, sem_acc, cm

if __name__ == "__main__":
    # Preset Pfad zur Main-Daten CSV
    input_csv = MAIN_CSV
    
    # Preset Hyperparameter
    n_estimators = 1000

    mean_acc, sem_acc, cm = run_loocv(input_csv, n_estimators=n_estimators)