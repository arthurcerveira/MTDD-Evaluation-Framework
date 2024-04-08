# Evaluation Framework for AI-driven Molecular Design of Multi-target Drugs

This repository contains the code and data for the paper "Evaluation Framework for AI-driven Molecular Design of Multi-target Drugs: Brain Diseases as a Case Study".

## Instructions

### Setup

To install the required packages, run the following command:

```bash
$ pip install -r requirements.txt
```

To download the required data and train the QSAR models for the brain diseases case study, run the following commands:

```bash
$ python scripts/external_data_sources.py
$ python scripts/qsar_pipeline.py
```

### Running the experiments

1. Build the dataset for the de Novo Design experiment:

```bash
$ python guacamol/data/get_data.py --holdout holdout_set_gcm_multitarget.smiles --destination guacamol/data/
```

2. (Optional) Compute the top K molecules for the de Novo Design experiment:

```bash
$ python scripts/top_k.py
```

The top K molecules are stored in the `guacamol/data/top_k` folder and save time when running the de Novo Design models over the benchmarks.

3. Run the de Novo Design models over the benchmarks:

```bash
$ python scripts/assess_baselines.py
```

The results of the experiments are saved in the `reports` folder. We can evaluate the QSAR models on the lead optimization task by running the following command:

```bash
$ python scripts/lo_task_benchmark.py
```