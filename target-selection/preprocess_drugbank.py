# import libraries to process xml
import xml.etree.ElementTree as ET
import json
from pprint import pprint
import pandas as pd

# open full database.xml file
tree = ET.parse('full-database.xml')
root = tree.getroot()

# Count the number of compounds in the database
count = 0

for drug in root:
    count += 1

print(f"There are {count:,} compounds on the database.")

def process_targets(targets):
    output = list()

    for target in targets:
        output.append({
            'name': target.find('{http://www.drugbank.ca}name').text,
            'organism': target.find('{http://www.drugbank.ca}organism').text,
            'actions': [action.text for action in target.find('{http://www.drugbank.ca}actions')],
            'known-action': target.find('{http://www.drugbank.ca}known-action').text,        
        })

    return output

# get all children of root of attrib type 'small molecule'
small_molecules = dict()
keys = [
    "drugbank-id",
    "name",
    "description",
    "toxicity",
    "protein-binding",
    "indication"
]

list_keys = [
    ("enzymes", "name"),
]

# Properties from calculated-properties
properties_keys = [
    "SMILES",
    "Molecular Formula",
    "InChI"
]

def child_to_dict(child):
    output = dict()

    for key in keys:
        if child.find('{http://www.drugbank.ca}' + key) is not None:
            output[key] = child.find('{http://www.drugbank.ca}' + key).text
        else:
            output[key] = None
    
    for key, subkey in list_keys:
        output[key] = list()
        child_element = child.find('{http://www.drugbank.ca}' + key)

        if child_element.text is not None:
            output[key].append(child_element.text)

        for subchild in child_element:
            name = subchild.find('{http://www.drugbank.ca}' + subkey)

            if name is not None:
                output[key].append(name.text)

    targets = child.find('{http://www.drugbank.ca}targets')
    if targets is not None:
        output['targets'] = process_targets(targets)
    else:
        output['targets'] = None

    properties = child.find('{http://www.drugbank.ca}calculated-properties')
    for property in properties:
        if property.find('{http://www.drugbank.ca}kind') is None:
            continue

        kind = property.find('{http://www.drugbank.ca}kind').text

        if kind not in properties_keys:
            continue

        value = property.find('{http://www.drugbank.ca}value').text
        output[kind] = value

    return output


for child in root:
    if child.attrib['type'] != 'small molecule':
        continue

    name = child.find('{http://www.drugbank.ca}name').text
    child_dict = child_to_dict(child)

    if child_dict['targets'] is None or len(child_dict['targets']) == 0:
        continue

    if child_dict['indication'] is None:
        continue

    if "SMILES" not in child_dict:
        continue

    small_molecules[name] = child_dict

with open('small_molecules.json', 'w') as outfile:
    json.dump(small_molecules, outfile, indent=4)
