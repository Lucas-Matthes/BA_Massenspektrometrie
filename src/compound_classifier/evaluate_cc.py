import os
import pandas as pd
from src.compound_classifier import CompoundClassifier
import joblib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from src.paths import NOVEL_CSV
from src.paths import COMPOUND_MODEL

# CC Laden
def load_cc():
    return joblib.load(COMPOUND_MODEL)

def evaluate_single_row(row, cc):

    features = row.drop(['molekuelname', 'category']).values
    name = row['molekuelname']
    true_cat = row['category']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pred_label = cc.predict_single(features)
        
    if (true_cat == 1 and pred_label == "outlier") or (true_cat in [2,3] and pred_label != "outlier"):
        correct = False
    else:
        correct = True

    print(f"Name: {name}, Kategorie: {true_cat}, Classified as: {pred_label}, Correct: {correct}")
    return name, true_cat, pred_label, correct

def evaluate_all(plot_first2_features=True):
    
    cc = load_cc()
    df = pd.read_csv(NOVEL_CSV)

    category_errors = {1: [], 2: [], 3: []}
    n_total = len(df)
    n_correct_inlier_outlier = 0
    n_inlier_total = 0
    n_inlier_correct_rf = 0

    # für Plot
    X_plot = []
    colors = []

    for _, row in df.iterrows():
        name, true_cat, pred_label, correct = evaluate_single_row(row, cc)

        # Inlier/Outlier Accuracy
        if (true_cat == 1 and pred_label != "outlier") or (true_cat in [2,3] and pred_label == "outlier"):
            n_correct_inlier_outlier += 1


        # RF Accuracy innerhalb Inlier
        if true_cat == 1:  # nur echte Inlier
            n_inlier_total += 1
            if pred_label != "outlier":  # korrekt als Inlier erkannt
                if pred_label == row['molekuelname']:  # Voting korrekt
                    n_inlier_correct_rf += 1


        # Fehlerliste pro Kategorie
        if not correct:
            category_errors[true_cat].append(name)

        # Plot vorbereiten
        if plot_first2_features:
            features = row.drop(['molekuelname', 'category']).values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                vote = cc.rf.predict_proba(features.reshape(1,-1))[0]  # Voting vector

            X_plot.append(vote[:2])  # nur die ersten beiden Dimensionen des Voting-Vektors

            if pred_label == "outlier":
                colors.append('black')
            elif correct:
                colors.append('green')
            else:
                colors.append('red')


    # Kennzahlen
    inlier_outlier_acc = n_correct_inlier_outlier / n_total if n_total > 0 else 0
    rf_inlier_acc = n_inlier_correct_rf / n_inlier_total if n_inlier_total > 0 else 0

    print("\n=== Zusammenfassung ===")
    print(f"Gesamt Samples: {n_total}")
    print(f"Inlier/Outlier Accuracy: {inlier_outlier_acc:.3f}")
    print(f"RF Accuracy innerhalb Inlier: {rf_inlier_acc:.3f}\n")
    for cat, errors in category_errors.items():
        print(f"Kategorie {cat} - Falsch gelabelte Moleküle ({len(errors)}): {errors}")

    # Optionaler Plot
    if plot_first2_features and X_plot:
        X_plot = np.array(X_plot)
        plt.figure(figsize=(8,6))
        plt.scatter(X_plot[:,0], X_plot[:,1], c=colors, edgecolor='k', s=80)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Novel Data Predictions (first 2 features)")
        plt.xlim(0, 1) 
        plt.ylim(0, 1) 
        plt.show()

    return category_errors, inlier_outlier_acc, rf_inlier_acc


if __name__ == "__main__":
    evaluate_all()
