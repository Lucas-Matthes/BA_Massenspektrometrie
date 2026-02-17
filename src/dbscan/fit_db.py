import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon
from kneed import KneeLocator
import joblib
import os
from src.paths import VECTORS_CSV
from src.paths import DBSCAN_MODEL


def auto_dbscan(
    data_csv = VECTORS_CSV, 
    model_file = DBSCAN_MODEL,
    metric = jensenshannon,
    min_inlier_ratio = 0.5,
    verbose = True
):
    # 1. CSV laden

    df = pd.read_csv(data_csv)
    molecule_names = df['molekuelname'].tolist()
    X = df.drop(columns=['molekuelname']).values

    n_samples, n_features = X.shape

    # 2. min_samples bestimmen

    min_samples = int(np.sqrt(n_samples / n_features))
    min_samples = max(min_samples, 2)

    if verbose:
        print(f"[auto_dbscan] min_samples = {min_samples}")


    # 3. k-Distanz-Kurve berechnen

    nbrs = NearestNeighbors(n_neighbors=min_samples, metric=metric).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])


    # 4. KneeLocator

    kl = KneeLocator(
        range(len(k_distances)),
        k_distances,
        curve='convex',
        direction='increasing'
    )

    eps_knee = k_distances[kl.knee] if kl.knee is not None else None

    if verbose:
        print(f"[auto_dbscan] knee eps = {eps_knee}")


    # 5. eps nach min_inlier_ratio

    eps_index = int(n_samples * min_inlier_ratio) - 1
    eps_minratio = k_distances[eps_index]

    if verbose:
        print(f"[auto_dbscan] min_inlier_ratio eps = {eps_minratio:.4f}")


    # 6. Hilfsfunktion zum Testen

    def test_eps(eps):
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        model.fit(X)
        inlier_ratio = np.mean(model.labels_ != -1)
        return model, inlier_ratio


    # 7. Knee-EPS testen

    if eps_knee is not None:
        model_knee, ratio_knee = test_eps(eps_knee)
        if verbose:
            print(f"[auto_dbscan] knee inlier ratio = {ratio_knee:.3f}")
    else:
        model_knee, ratio_knee = None, 0


    # 8. Entscheidung, welcher eps gewählt wird

    if eps_knee is not None and ratio_knee >= min_inlier_ratio:
        eps_final = eps_knee
        db = model_knee
        if verbose:
            print("[auto_dbscan] → knee eps akzeptiert")
    else:
        eps_final = eps_minratio
        db, _ = test_eps(eps_final)
        if verbose:
            print("[auto_dbscan] → knee eps verworfen, min_inlier_ratio eps gewählt")

    if verbose:
        print(f"[auto_dbscan] FINAL eps = {eps_final:.4f}")


    # 9. Modell speichern

    model_dir = os.path.dirname(model_file)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_data = {
        "dbscan": db,
        "eps": eps_final,
        "eps_knee": eps_knee,
        "eps_minratio": eps_minratio,
        "min_samples": min_samples,
        "molecule_names": molecule_names,
        "metric": metric
    }

    joblib.dump(model_data, model_file)

    if verbose:
        print(f"[auto_dbscan] Modell gespeichert in: {model_file}")

    return model_data

if __name__ == "__main__":
    auto_dbscan(
        min_inlier_ratio=0.5,
    )
