import os

from src.data_extraction.extract_main import process_folder as process_main
from src.data_extraction.extract_novel import process_folder as process_novel
from src.random_forest.build_rf import build_rf_model
from src.random_forest.rfLoocv import run_loocv
from src.random_forest.vectorize import vectorize_rf
from src.dbscan.fit_db import auto_dbscan
from src.dbscan.evaluate_db import evaluate_db
from src.compound_classifier.build_cc import build_compound_classifier
from src.compound_classifier.evaluate_cc import evaluate_all
from src.paths import DATA_DIR
from src.paths import MODELS_DIR
from src.paths import OUTPUT_DIR

class MasterPipeline:
    def __init__(
        self,
        HO_run=False,
        bin_array=None,
        peak_array=None,
        estimator_array=None,
        sheet="200",
        count=10,
        bin_size=1.0,
        sort_method="int",
        n_estimators=500,
        extract_main=True,
        extract_novel=True,
        eval_rf=False,
        build_rf=True,
        vectorize=True,
        build_db=True,
        min_inlier_ratio=0.95,
        eval_db=False,
        build_cc=True,
        eval_cc=False
    ):
        self.HO_run = HO_run
        self.bin_array = bin_array if bin_array is not None else [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.peak_array = peak_array if peak_array is not None else [1, 2, 5, 10, 15, 20, 25]
        self.estimator_array = estimator_array if estimator_array is not None else [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        self.sheet = sheet
        self.count = count
        self.bin_size = bin_size
        self.sort_method = sort_method
        self.n_estimators = n_estimators
        self.extract_main = extract_main
        self.extract_novel = extract_novel
        self.eval_rf = eval_rf
        self.build_rf = build_rf
        self.vectorize = vectorize
        self.build_db = build_db
        self.min_inlier_ratio = min_inlier_ratio
        self.eval_db = eval_db
        self.build_cc = build_cc
        self.eval_cc = eval_cc

        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.models_dir = MODELS_DIR

    def run(self):
        if self.HO_run:
            print("Running Hyperparameter Looper instead of full pipeline...")
            from random_forest.looper import sweep_estimators, heatmap_peaks_bins
            # Beispielaufrufe für HO, evtl Parameter weiterreichen
            sweep_estimators(sheet=self.sheet, peak_count=self.count, bin_size=self.bin_size, sort_method=self.sort_method, estimators=self.estimator_array)
            heatmap_peaks_bins(sheet=self.sheet, n_estimators=50, sort_method=self.sort_method, peak_counts=self.peak_array, bin_sizes=self.bin_array)
            return

        # 1 Data Extraction
        if self.extract_main or self.extract_novel:
            print("Step 1: Data Extraction")
            if self.extract_main:
                process_main(
                    input_folder = os.path.join(self.data_dir, "main"),
                    output_csv = os.path.join(self.output_dir, "main.csv"),
                    count=self.count,
                    bin_size=self.bin_size,
                    sheet=self.sheet,
                    sort_method=self.sort_method
                    )

            if self.extract_novel:
                process_novel(
                    input_folder = os.path.join(self.data_dir, "novel"),
                    output_csv = os.path.join(self.output_dir, "novel.csv"),
                    count=self.count,
                    bin_size=self.bin_size,
                    sheet=self.sheet,
                    sort_method=self.sort_method
                    )

        # 2 Random Forest
        if self.eval_rf or self.build_rf:
            print("Step 2: Random Forest")
            rf_model_path = os.path.join(self.models_dir, "rf_model.joblib")
            if self.build_rf:
                build_rf_model(input_csv=os.path.join(self.output_dir, "main.csv"), output_model_path=rf_model_path, n_estimators=self.n_estimators)
            if self.vectorize:
                vector_csv_path = os.path.join(self.models_dir, "vectors.csv")
                vectorize_rf(rf_model_path=rf_model_path, output_csv_path=vector_csv_path, main_csv_path=os.path.join(self.output_dir, "main.csv"))
            if self.eval_rf:
                run_loocv(input_csv=os.path.join(self.output_dir, "main.csv"), n_estimators=self.n_estimators)

        # 3 DBSCAN
        if self.build_db:
            print("Step 3: Fit DBSCAN")
            vectors_csv = os.path.join(self.models_dir, "vectors.csv")
            dbscan_model_path = os.path.join(self.models_dir, "dbscan_classifier.pkl")
            auto_dbscan(
                data_csv=vectors_csv,
                model_file=dbscan_model_path,
                min_inlier_ratio=self.min_inlier_ratio
            )
        if self.eval_db:
            print("Step 4: Evaluate DBSCAN")
            vectors_csv = os.path.join(self.models_dir, "vectors.csv")
            dbscan_model_path = os.path.join(self.models_dir, "dbscan_classifier.pkl")
            evaluate_db()

        # 4 Compound Classifier
        if self.build_cc:
            print("Step 5: Build Compound Classifier")
            build_compound_classifier()
        if self.eval_cc:
            print("Step 6: Evaluate Compound Classifier")
            evaluate_all()


if __name__ == "__main__":
    print("started")
    import sys
    import os

    # Parent folder zur PATH hinzufügen (damit 'src' als Paket funktioniert)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.MasterPipeline import MasterPipeline

    pipeline = MasterPipeline(
        eval_cc=True, eval_rf=False, eval_db=False, extract_novel=True,
        n_estimators = 1000, count=5, bin_size=0.05, min_inlier_ratio=0.9
    )
    pipeline.run()