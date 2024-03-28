from pprint import pprint
import json
import os
import pandas as pd
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
models_dir = current_dir.parent / "models"
results_dir = current_dir.parent / "reports"
sys.path.append(str(current_dir.parent))

from bambu.download import download_pubchem_assay_data
from bambu.preprocess import preprocess_lo_split
from bambu.train import train_lo_split
from bambu.validate import validate


BIOASSAY_IDS = {    
    "AChE": 1347395,
    "MAOB": None,
    "D2R": 485344,
    "_5HT2A": 624169,
    "D3R": 652048,
    "BBB": None,
}


def qsar_pipeline(target, assay_id):
    assay_file = data_dir / f"{target}.csv"

    if (assay_id is not None) and (not os.path.exists(assay_file)):
        download_pubchem_assay_data(
            pubchem_assay_id=assay_id, 
            output=assay_file, 
            pubchem_InchI_chunksize=100
        )

    preprocess_lo_split(
        input_file=data_dir / f"{target}.csv", 
        output_file_train=data_dir / f"{target}-Train-Preprocessed.csv",
        output_file_val=data_dir / f"{target}-Val-Preprocessed.csv",
        output_preprocessor_file=models_dir / "Morgan.pkl",
        feature_type="morgan-1024",
    )

    train_lo_split(
        input_train=data_dir / f"{target}-Train-Preprocessed.csv",
        input_val=data_dir / f"{target}-Val-Preprocessed.csv",
        output=models_dir / f"{target}.pkl",
        estimators=['rf', 'extra_tree', 'decision_tree',  # 'svm', 
                    'gradient_boosting', 'neural_network'],
        threads=7,
        time_budget=3600,
    )

    df_train = pd.read_csv(data_dir / f"{target}-Train-Preprocessed.csv")
    df_test = pd.read_csv(data_dir / f"{target}-Val-Preprocessed.csv")

    report = validate(
        input_train=df_train,
        input_test=df_test,
        model_path=models_dir / f"{target}.pkl",
        output=None,
    )

    return report["raw_scores"]


if __name__ == "__main__":
    validation_reports = dict()

    for target, assay_id in BIOASSAY_IDS.items():
        print(f"Processing {target}...")
        validation_reports[target] = qsar_pipeline(target, assay_id)

    pprint(validation_reports)

    # Save report
    with open(results_dir / "QSAR-Models.json", "w") as f:
        json.dump(validation_reports, f, indent=4)
