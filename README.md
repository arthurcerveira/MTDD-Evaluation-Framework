# Evaluation Framework for AI-driven Molecular Design of Multi-target Drugs

This repository contains the code and data for the paper "[Evaluation Framework for AI-driven Molecular Design of Multi-target Drugs: Brain Diseases as a Case Study](https://arxiv.org/abs/2408.10482)".

## Instructions

### Setup

The code was written and tested in Python 3.11.6. To install the required packages, run the following command:

```bash
$ pip install -r requirements.txt
```

To download the required data and train the QSAR models for the brain diseases case study, run the following commands:

```bash
$ python scripts/external_data_sources.py
$ python scripts/qsar_pipeline.py
```

### Target selection

The target selction process is described in [`target-selection/README.md`](target-selection/README.md).

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

## Citation

If you find this project useful for your research, please consider citing the following BibTeX entry.

```bibtex
@INPROCEEDINGS{10611839,
  author={Cerveira, Arthur and Kremer, Frederico and Lourenço, Darling and Corrêa, Ulisses B.},
  booktitle={2024 IEEE Congress on Evolutionary Computation (CEC)}, 
  title={Evaluation Framework for AI -driven Molecular Design of Multi-target Drugs: Brain Diseases as a Case Study}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Drugs;Training;Toxicology;Biological system modeling;Modulation;Evolutionary computation;Benchmark testing;molecular design;multi-target drug discovery;evolutionary algorithms;deep generative models;de novo design},
  doi={10.1109/CEC60901.2024.10611839}}
```