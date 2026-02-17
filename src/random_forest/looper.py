import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_extraction.extract_main import process_folder
from random_forest.rfLoocv import run_loocv
from src.paths import DATA_DIR
from src.paths import MAIN_CSV


def sweep_estimators(sheet="200", peak_count=5, bin_size=0.05, sort_method="int", estimators=[10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]):
    """
    Plottet Accuracy vs n_estimators für feste Sheet, Peak Count und Bin Size.
    """
    accuracies, sems = [], []

    # Main-Daten einml erzeugen
    process_folder(DATA_DIR, MAIN_CSV, count=peak_count, bin_size=bin_size, sheet=sheet, sort_method=sort_method)

    for n in estimators:
        mean_acc, sem_acc = run_loocv(MAIN_CSV, n_estimators=n)
        accuracies.append(mean_acc)
        sems.append(sem_acc)

    # x-Positionen gleichmäßig verteilen
    x_idx = np.arange(len(estimators))

    plt.errorbar(x_idx, accuracies, yerr=sems, fmt='-o')
    plt.xticks(x_idx, estimators)  # echte Werte als Labels
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. n_estimators ({sheet} eV, {peak_count} peaks, bin={bin_size}, {sort_method})")
    plt.ylim(0, 1)
    plt.show()


def heatmap_peaks_bins(sheet="200", n_estimators=50, peak_counts=[1,3,5,10,20], bin_sizes=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5], sort_method="int"):
    """
    Brute-Force Heatmap: Accuracy für alle Kombinationen von Peak Count und Bin Size.
    """
    results = np.zeros((len(peak_counts), len(bin_sizes)))

    for i, p in enumerate(peak_counts):
        for j, b in enumerate(bin_sizes):
            # Main-Daten CSV erzeugen
            process_folder(DATA_DIR, MAIN_CSV, count=p, bin_size=b, sheet=sheet, sort_method=sort_method)
            mean_acc, _ = run_loocv(MAIN_CSV, n_estimators=n_estimators)
            results[i, j] = mean_acc
            print(f"Peak Count={p}, Bin Size={b}: Accuracy={mean_acc:.3f}")

    plt.figure(figsize=(8,6))
    sns.heatmap(results, annot=True, xticklabels=bin_sizes, yticklabels=peak_counts, cmap="viridis")
    plt.xlabel("Bin Size")
    plt.ylabel("Peak Count")
    plt.title(f"Accuracy Heatmap ({sheet} eV, n_estimators={n_estimators}, sort={sort_method})")
    plt.show()


if __name__ == "__main__":
    # Beispielaufrufe:
    sweep_estimators()
    heatmap_peaks_bins(sheet="200", n_estimators=100, sort_method="int")
