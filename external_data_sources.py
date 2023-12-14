import os

import pandas as pd
from bambu.download import download_pubchem_assay_data
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

BBB_URI = "https://raw.githubusercontent.com/omixlab/alzheimer-drug-ml/main/data/raw/B3DB.csv"


def smiles_to_inchi(smiles, max_mols=10_000):
    mols = list()
    
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            continue

        if mol is None:
            continue

        inchi = Chem.MolToInchi(mol)
        mols.append(inchi)

        if len(mols) >= max_mols:
            break       

    return mols


def get_zinc_dataset(output="data/ZINC.csv"):
    zinc_complete = pd.read_csv(
        "data/ZINC-Complete.txt", sep="\t", low_memory=False
    ).sample(frac=1.0)  # Shuffle the data
    
    # Convert smiles to InChI
    inchis = smiles_to_inchi(zinc_complete["smiles"], max_mols=1_000_000)
    zinc = pd.DataFrame({
        "InChI": inchis,
        # Zinc is used as a source of inactive examples
        "activity": "inactive"
    })

    zinc.to_csv(output, index=False)


def get_nrtk3_assays(output="data/NTRK3.csv"):
    # Download assays from PubChem
    nrtk3 = [
        "1229326",
        "1741214",
        "1822411",
        "1195217"
    ]

    for assay_id in nrtk3:
        download_pubchem_assay_data(
            pubchem_assay_id=assay_id, 
            output=f"data/{assay_id}.csv", 
            pubchem_InchI_chunksize=100
        )

    # Combine all the data into one file
    nrtk3_active = pd.concat([
        pd.read_csv(f"data/{assay_id}.csv") for assay_id in nrtk3
    ])

    # Delete temporary files
    for assay_id in nrtk3:
        os.remove(f"data/{assay_id}.csv")

    # Enrich with inactive examples
    zinc = pd.read_csv("data/ZINC.csv")

    nrtk3_inactive_lenght = len(nrtk3_active)
    nrtk3_mols = [Chem.MolFromInchi(inchi) for inchi in nrtk3_active["InChI"]]
    nrtk3_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in nrtk3_mols]

    inactive_molecules = list()

    for inchi in zinc["InChI"]:
        mol = Chem.MolFromInchi(inchi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)

        scores = np.array(DataStructs.BulkTanimotoSimilarity(fp, nrtk3_fps))

        if ((scores >= 0.3) & (scores < 0.7)).any():
            inactive_molecules.append(inchi)

        if len(inactive_molecules) >= nrtk3_inactive_lenght:
            break

    nrtk3_inactive = pd.DataFrame({
        "InChI": inactive_molecules,
        "activity": "inactive"
    })

    nrtk3_assays = pd.concat([nrtk3_active, nrtk3_inactive])

    nrtk3_assays = nrtk3_assays.drop_duplicates(
        subset=["InChI"]
    )

    nrtk3_assays.to_csv(output, index=False)


def get_nrtk1_assays(output="data/NTRK1.csv"):
    print("Building NTRK1 dataset...")
    nrtk1_id = "1802770"

    download_pubchem_assay_data(
        pubchem_assay_id=nrtk1_id, 
        output=f"data/{nrtk1_id}.csv", 
        pubchem_InchI_chunksize=100
    )

    # Enrich with inactive examples
    nrtk1_active = pd.read_csv(f"data/{nrtk1_id}.csv")
    zinc = pd.read_csv("data/ZINC.csv").sample(frac=1.0)

    nrtk1_inactive_lenght = len(nrtk1_active)
    nrtk1_mols = [Chem.MolFromInchi(inchi) for inchi in nrtk1_active["InChI"]]
    nrtk1_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in nrtk1_mols]

    inactive_molecules = list()

    # for inchi in zinc["InChI"]:
    for inchi in tqdm(zinc["InChI"]):
        try:
            mol = Chem.MolFromInchi(inchi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        except:
            continue

        scores = np.array(DataStructs.BulkTanimotoSimilarity(fp, nrtk1_fps))

        if ((scores >= 0.3) & (scores < 0.7)).any():
            inactive_molecules.append(inchi)

        if len(inactive_molecules) >= nrtk1_inactive_lenght:
            print("Found enough inactive molecules")
            break

    nrtk1_inactive = pd.DataFrame({
        "InChI": inactive_molecules,
        "activity": "inactive"
    })

    print("Active molecules:", len(nrtk1_active))
    print("Inactive molecules:", len(inactive_molecules))

    nrtk1_assays = pd.concat([nrtk1_active, nrtk1_inactive])

    # Delete temporary files
    os.remove(f"data/{nrtk1_id}.csv")

    nrtk1_assays = nrtk1_assays.drop_duplicates(
        subset=["InChI"]
    )

    nrtk1_assays.to_csv(output, index=False)


def get_bbb_dataset(output="data/BBB.csv"):
    bbb = pd.read_csv(BBB_URI)
    bbb = bbb[["InChI", "activity"]]

    # Filter out corrupted mols
    bbb_filter = bbb.copy()
    bbb_filter["corrupted"] = bbb_filter["InChI"].apply(
        lambda m: True if Chem.inchi.MolFromInchi(m) is None else False
    )

    bbb_filter = bbb_filter[~bbb_filter["corrupted"]]

    bbb_filter.to_csv(output, index=False)


def get_ros1_dataset(output="data/ROS1.csv"):
    ros1 = pd.read_csv("data/ROS1-ChEMBL.csv", sep=";")
    inchis = smiles_to_inchi(ros1["Smiles"])
    activities = ros1["Standard Type"].apply(
        lambda x: "active" if x.lower() == "inhibition" else "inactive"
    )

    ros1 = pd.DataFrame({
        "InChI": inchis,
        "activity": activities
    })    

    ros1.to_csv(output, index=False)


if __name__ == "__main__":
    get_zinc_dataset()
    get_nrtk3_assays()
    get_nrtk1_assays()
    get_bbb_dataset()
    get_ros1_dataset()