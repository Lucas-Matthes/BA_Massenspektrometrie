import joblib
import numpy as np

class CompoundClassifier:
    def __init__(self, rf_path, dbscan_path):
        self.rf = joblib.load(rf_path)                     # Random Forest
        db_dict = joblib.load(dbscan_path)                # DBSCAN-Datenstruktur
        
        self.db = db_dict["dbscan"]
        self.db_eps = db_dict["eps"]
        self.db_min_samples = db_dict["min_samples"]
        self.molecule_names = db_dict["molecule_names"]   # Reihenfolge der RF-Klassen
        
        self.class_order = self.rf.classes_               # RF-Label → Index Map

    def _ensure_vector(self, x):
        if isinstance(x, (list, tuple)):
            return np.array(x)
        if isinstance(x, dict):
            return np.array(list(x.values()))
        if hasattr(x, "values"):
            return x.values
        return x

    def predict_single(self, x):
        x = self._ensure_vector(x).reshape(1, -1)

        # 1) RF -> Voting probability vector
        vote = self.rf.predict_proba(x)[0]

        # 2) DBSCAN -> Inlier / Outlier Test
        dist = np.linalg.norm(self.db.components_ - vote, axis=1)
        is_inlier = np.any(dist <= self.db_eps)

        if not is_inlier:
            return "outlier"

        # 3) Rückgabe -> RF Popular Vote
        idx = np.argmax(vote)
        return self.class_order[idx]

    def predict_batch(self, X):
        return [self.predict_single(x) for x in X]