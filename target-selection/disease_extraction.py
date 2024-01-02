import json
from datetime import datetime
import os

from llm_utils import (
    run_query, parser, llm, prompt_template
)


with open('small_molecules.json') as f:
    data = json.load(f)

mols_lenght = len(data)

drug_disease = dict()

# Load existing data if it exists
if os.path.exists('drug_disease.json'):
    with open('drug_disease.json') as f:
        drug_disease = json.load(f)

for index, drug in enumerate(data):
    print(f"[{datetime.now():%H:%M:%S}] {index+1}/{mols_lenght} {drug}")

    if drug in drug_disease:
        print(f"Already in drug_disease.json, skipping...")
        continue

    metadata = data[drug]
    # print(metadata['indication'], '\n')

    output = run_query(metadata['indication'], prompt_template, llm, parser)
    print(output, '\n')

    drug_disease[drug] = output['disease']

    # Save after each iteration
    with open('drug_disease.json', 'w') as f:
        json.dump(drug_disease, f, indent=4)
