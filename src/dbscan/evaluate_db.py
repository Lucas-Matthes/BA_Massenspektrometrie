import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import DBSCAN
from itertools import combinations
from src.paths import VECTORS_CSV
from src.paths import DBSCAN_MODEL
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import jensenshannon

def evaluate_db(vectors_csv=VECTORS_CSV,
                dbscan_model_path=DBSCAN_MODEL,
                show_plots=True):

    # Load DBSCAN dictionary
    dbscan_dict = joblib.load(dbscan_model_path)
    dbscan = dbscan_dict["dbscan"]
    eps = dbscan_dict["eps"]
    min_samples = dbscan_dict["min_samples"]
    molecule_names_saved = dbscan_dict["molecule_names"]
    metric = dbscan_dict["metric"]

    # Load voting vectors
    df = pd.read_csv(vectors_csv)
    molecule_names = df["molekuelname"]
    X = df.drop(columns=["molekuelname"]).values

    # Labels aus der bereits gefitteten DBSCAN Instanz
    labels = dbscan.labels_
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    n_outliers = np.sum(labels == -1)
    n_inliers = len(labels) - n_outliers
    inlier_ratio = n_inliers / len(labels)

    print("=== DBSCAN Evaluation ===")
    print(f"Total samples: {len(labels)}")
    print(f"Number of clusters (excluding outliers): {n_clusters}")
    print(f"Number of outliers: {n_outliers}")
    print(f"Inlier ratio: {inlier_ratio:.3f}")
    print(f"min_samples: {min_samples}, eps: {eps}")

    # Cluster -> Moleküle mapping
    cluster_mapping = {}
    for lbl in unique_labels:
        cluster_mapping[lbl] = molecule_names[labels == lbl].tolist()
        if lbl != -1:
            print(f"\nCluster {lbl} ({len(cluster_mapping[lbl])} samples): {cluster_mapping[lbl]}")

    # Maximaler Eps-Abstand innerhalb von Clustern
    print("\nMaximaler Eps-Abstand pro Cluster:")
    for lbl in unique_labels:
        if lbl == -1:
            continue
        cluster_points = X[labels == lbl]
        if len(cluster_points) < 2:
            max_dist = 0
        else:
            max_dist = np.max([metric(a,b) for a,b in combinations(cluster_points, 2)])
        print(f"Cluster {lbl}: max distance = {max_dist:.3f}")

    if show_plots:
        # Scatterplot der ersten zwei Dimensionen
        plt.figure(figsize=(8,6))
        for lbl in unique_labels:
            points = X[labels == lbl]
            if lbl == -1:
                plt.scatter(points[:,0], points[:,1], c='k', marker='x', label="Outliers")
            else:
                plt.scatter(points[:,0], points[:,1], label=f"Cluster {lbl}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("DBSCAN Clusters (first 2 features)")
        plt.legend()
        plt.show()

        # Cluster Größen Barplot
        cluster_sizes = [np.sum(labels==lbl) for lbl in unique_labels if lbl!=-1]
        plt.figure(figsize=(6,4))
        sns.barplot(x=list(range(n_clusters)), y=cluster_sizes)
        plt.xlabel("Cluster ID")
        plt.ylabel("Cluster Size")
        plt.title("Cluster Sizes")
        plt.show()

    # Cluster Label-Verteilung pro Cluster
    print("\nCluster label distributions:")
    for lbl in unique_labels:
        if lbl == -1:
            continue
        cluster_labels = molecule_names[labels == lbl].value_counts()
        print(f"\nCluster {lbl}:")
        print(cluster_labels)

def cluster_overlap_finder(vectors_csv=VECTORS_CSV):
    df = pd.read_csv(vectors_csv)

    labels = df["molekuelname"].values
    X = df.drop(columns=["molekuelname"]).values

    def jsd_distance_matrix(X):
        return squareform(pdist(X, metric=lambda u,v: jensenshannon(u,v)))

    D = jsd_distance_matrix(X)


    # 3. DBSCAN mit precomputed distanc

    db = DBSCAN(
        eps=0.15,       
        min_samples=3,
        metric="precomputed"
    )

    cluster_ids = db.fit_predict(D)

    
    # 4. Ergebnis DataFrame
    
    result = pd.DataFrame({
        "label": labels,
        "cluster": cluster_ids
    })

    print("\nCluster Zuordnung:")
    print(result)

    
    # 5. Prüfen ob Cluster Labels mischen (Overlap!)
    
    print("\n==============================")
    print("Cluster Overlap Analyse")
    print("==============================")

    clusters = sorted(result["cluster"].unique())

    overlapping_clusters = []

    for c in clusters:
        if c == -1:
            continue  # DBSCAN noise ignorieren
        
        cluster_points = result[result["cluster"] == c]
        unique_labels = cluster_points["label"].unique()

        print(f"\nCluster {c}:")
        print(" Punkte:", len(cluster_points))
        print(" Labels:", list(unique_labels))

        if len(unique_labels) > 1:
            print(" >>> OVERLAP DETECTED!")
            overlapping_clusters.append(c)

    
    # 6. Globales Ergebnis
    print("\n==============================")
    if overlapping_clusters:
        print("Überlappende Cluster gefunden:", overlapping_clusters)
    else:
        print("Keine Clusterüberlappung")

if __name__ == "__main__":
    # evaluate_db()
    cluster_overlap_finder()