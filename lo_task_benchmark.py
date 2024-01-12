import pickle
import json

import lohi_splitter as lohi
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)

from qsar_pipeline import BIOASSAY_IDS


def load_bioassay_dataframe(target):
    assay_file = f"data/{target}.csv"

    df = pd.read_csv(assay_file)
    df = df.dropna(subset=["InChI"])
    df = df.drop_duplicates(subset=["InChI"])

    # Undersample the majority class before splitting into LO train/test
    rus = RandomUnderSampler(random_state=42)
    df, _ = rus.fit_resample(df, df["activity"])
        
    inchi_to_smiles = lambda x: Chem.MolToSmiles(Chem.inchi.MolFromInchi(x))

    df["smiles"] = df["InChI"].apply(lambda x: inchi_to_smiles(x) if Chem.inchi.MolFromInchi(x) is not None else None)
    df = df.dropna(subset=["smiles"])
    df["activity"] = df["activity"].apply(lambda x: 1 if x == "active" else 0)
    df["values"] = df["activity"]

    print("Dataframe shape:", df.shape)

    return df


def split_lohi(df, threshold=0.323, min_cluster_size=5, max_clusters=50, std_threshold=0.1):
    smiles = df['smiles'].to_list()
    values = df['values'].to_list()

    # Similarity threshold for clustering molecules.
    # Molecules are considered similar if their ECFP4 Tanimoto Similarity is larger than this threshold.
    threshold=threshold

    # The minimum number of molecules required in a cluster.
    min_cluster_size=min_cluster_size

    # Maximum number of clusters to be created. Any additional molecules are added to the training set.
    max_clusters=max_clusters

    # Minimum standard deviation of values within a cluster.
    # This ensures that clusters with too little variability are filtered out.
    # For further details, refer to Appendix B of the paper.
    std_threshold=std_threshold

    cluster_smiles, train_smiles = lohi.lo_train_test_split(smiles=smiles, 
                                                    threshold=threshold, 
                                                    min_cluster_size=min_cluster_size, 
                                                    max_clusters=max_clusters, 
                                                    values=values, 
                                                    std_threshold=std_threshold)

    # It is just handy util to build pandas DataFrame
    split = lohi.set_cluster_columns(df, cluster_smiles, train_smiles)

    train = split[split['cluster'] == 0]
    test = split[split['cluster'] != 0]

    return train, test


def get_lo_metrics(data, y_pred):
    data = data.copy()
    data['preds'] = y_pred

    y_true_list = []
    y_pred_list = []

    for cluster_idx in data['cluster'].unique():
        cluster = data[data['cluster'] == cluster_idx]

        y_true_list.append(cluster['values'])
        y_pred_list.append(cluster['preds'])

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred)
    }

    return metrics


def fit_predict(train, test, model):
    train_mols = [Chem.MolFromSmiles(x) for x in train['smiles']]
    train_morgan_fps = np.array(
        [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_mols]
    )

    test_mols = [Chem.MolFromSmiles(x) for x in test['smiles']]
    test_morgan_fps = np.array(
        [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in test_mols]
    )

    fitted_model = model.fit(train_morgan_fps, train['values'])

    train_result = train.copy()
    train_result['preds'] = fitted_model.predict(train_morgan_fps)

    test_result = test.copy()
    test_result['preds'] = fitted_model.predict(test_morgan_fps)

    return train_result, test_result


if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*')
    results = dict()

    for target in BIOASSAY_IDS:
        print(target)
        df = load_bioassay_dataframe(target)

        print("Splitting into LO train/test...")
        train, test = split_lohi(df)

        print("Train shape:", train.shape)
        print("Test shape:", test.shape)

        if len(test) == 0 or train['activity'].nunique() < 2:
            print("LO split failed")
            continue

        print("Unique train classes:\n", train['activity'].value_counts())
        print("\nUnique test classes:\n", test['activity'].value_counts())

        print("Fitting the model...")
        model_path = f"models/{target}.pkl"
        with open(model_path, 'rb') as f:
            automl = pickle.load(f)

        model = automl.model.estimator

        # Print the model class
        print(model)

        train_result, test_result = fit_predict(train, test, model)

        print("Evaluating...")
        train_metrics = get_lo_metrics(train_result, train_result['preds'])
        test_metrics = get_lo_metrics(test_result, test_result['preds'])

        print(f"{target} train: {train_metrics}")
        print(f"{target} test: {test_metrics}")
        print()

        results[target] = {
            **test_metrics,
            'model': model.__class__.__name__
        }

    with open(f"reports/Lo-Task-Benchmarks.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save as CSV
    df = pd.DataFrame(index=results.keys(), data=results.values())
    df.to_csv("reports/Lo-Task-Benchmarks.csv")
