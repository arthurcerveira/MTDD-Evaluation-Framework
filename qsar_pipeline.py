from pprint import pprint
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from bambu.download import download_pubchem_assay_data
from bambu.preprocess import preprocess
from bambu.train import train
from bambu.validate import validate


BIOASSAY_IDS = {    
    "AChE": 1347395,
    "MAOB": None,
    "D2R": 485344,
    "_5HT2A": 624169,
    "D3R": 652048,
    "BBB": None,
    # "APP": 1276,
    # "NTRK3": None,
    # "NTRK1": None,
    # "ROS1": None,
    # "D4R": None,
}


def qsar_pipeline(target, assay_id):
    assay_file = f"data/{target}.csv"

    if (assay_id is not None) and (not os.path.exists(assay_file)):
        download_pubchem_assay_data(
            pubchem_assay_id=assay_id, 
            output=assay_file, 
            pubchem_InchI_chunksize=100
        )

    preprocess(
        input_file=f"data/{target}.csv", 
        output_file=f"data/{target}-Preprocessed.csv",
        output_preprocessor_file="models/Morgan.pkl",
        feature_type="morgan-1024",
        undersample=True,
    )

    train(
        input_train=f"data/{target}-Preprocessed.csv",
        output=f"models/{target}.pkl",
        estimators=['rf', 'extra_tree', 'decision_tree', 'svm', 
                    'logistic_regression', 'gradient_boosting'],
        threads=15,
        max_iter=100,
    )

    df = pd.read_csv(f"data/{target}-Preprocessed.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    report = validate(
        input_train=df_train,
        input_test=df_test,
        model_path=f"models/{target}.pkl",
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
    with open(f"reports/QSAR-Models.json", "w") as f:
        json.dump(validation_reports, f, indent=4)
