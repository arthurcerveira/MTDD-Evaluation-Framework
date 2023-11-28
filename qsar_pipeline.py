from pprint import pprint
import json
import os

from bambu.download import download_pubchem_assay_data
from bambu.preprocess import preprocess
from bambu.train import train
from bambu.validate import validate
import pandas as pd
from sklearn.model_selection import train_test_split


BIOASSAY_IDS = {
    "APP": 1276,
    "AChE": 1347395,
    "BBB": None,
    "D2R": 485344,
    "_5HT1A": 624169,
    "NTRK3": None,
    "NTRK1": None
}

validation_report = dict()


for target, assay_id in BIOASSAY_IDS.items():
    print(f"Processing {target}...")
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
        feature_type="morgan-64",
        undersample=True,
    )

    train(
        input_train=f"data/{target}-Preprocessed.csv",
        output=f"models/{target}.pkl",
        estimators=['rf', 'extra_tree', 'decision_tree', 'svm', 
                    'logistic_regression', 'gradient_boosting'],
        threads=7,
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

    validation_report[target] = report["raw_scores"]

pprint(validation_report)

# Save report
with open(f"reports/QSAR-Models.json", "w") as f:
    json.dump(validation_report, f, indent=4)
