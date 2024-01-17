#%%
import os

from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from baselines.graph_ga.goal_directed_generation import GB_GA_Generator
from baselines.smiles_ga.goal_directed_generation import ChemGEGenerator
from baselines.smiles_lstm_ppo.goal_directed_generation import PPODirectedGenerator
from baselines.graph_mcts.goal_directed_generation import GB_MCTS_Generator
from baselines.random_smiles_sampler.goal_directed_generation import RandomSmilesSampler
from baselines.random_smiles_sampler.optimizer import RandomSamplingOptimizer
from baselines.smiles_lstm_hc.goal_directed_generation import SmilesRnnDirectedGenerator
from baselines.best_from_chembl.optimizer import BestFromChemblOptimizer
from baselines.best_from_chembl.chembl_file_reader import ChemblFileReader


SMILES_HTS_FILE = 'guacamol/data/guacamol_v1_50k.smiles'
N_JOBS = -1


#%%
print("Generating molecules with GB_GA_Generator")

optimiser = GB_GA_Generator(
    smi_file=SMILES_HTS_FILE,
    population_size=100,
    offspring_size=200,
    generations=1000,
    mutation_rate=0.01,
    patience=5,
    n_jobs=N_JOBS,
    random_start=False,
)

json_file_path = os.path.join("reports", 'graph_ga.json')

assess_goal_directed_generation(
    optimiser, json_output_file=json_file_path, benchmark_version="multitarget"
)

#%%
print("Generating molecules with ChemGEGenerator")

optimiser = ChemGEGenerator(
    smi_file=SMILES_HTS_FILE,
    population_size=100,
    n_mutations=200,
    gene_size=300,
    generations=1000,
    n_jobs=N_JOBS,
    random_start=False,
    patience=5
)

json_file_path = os.path.join("reports", 'smiles_ga.json')

assess_goal_directed_generation(
    optimiser, json_output_file=json_file_path, benchmark_version="multitarget"
)

#%%
print("Generating molecules with RandomSmilesSampler")

with open(SMILES_HTS_FILE, 'r') as smiles_file:
    smiles_list = smiles_file.readlines()

sampler = RandomSmilesSampler(molecules=smiles_list)
optimiser = RandomSamplingOptimizer(sampler=sampler)

json_file_path = os.path.join("reports", 'random_smiles.json')

assess_goal_directed_generation(
    optimiser, json_output_file=json_file_path, benchmark_version="multitarget"
)

#%%
print("Generating molecules with BestFromChemblOptimizer")

smiles_reader = ChemblFileReader(SMILES_HTS_FILE)
optimizer = BestFromChemblOptimizer(smiles_reader=smiles_reader, n_jobs=N_JOBS)

json_file_path = os.path.join("reports", 'best_from_chembl.json')

assess_goal_directed_generation(optimizer, json_output_file=json_file_path, benchmark_version="multitarget")

#%%
print("Generating molecules with SmilesRnnDirectedGenerator")

model_path = os.path.join("baselines/smiles_lstm_hc", 'pretrained_model', 'model_final_0.473.pt')
optimizer = SmilesRnnDirectedGenerator(pretrained_model_path=model_path,
                                        n_epochs=20,
                                        mols_to_sample=1024,
                                        keep_top=512,
                                        optimize_n_epochs=2,
                                        max_len=100,
                                        optimize_batch_size=256,
                                        number_final_samples=4096,
                                        random_start=False,
                                        smi_file=SMILES_HTS_FILE,
                                        n_jobs=N_JOBS)

json_file_path = os.path.join("reports", 'smiles_lstm_hc.json')
assess_goal_directed_generation(optimizer, json_output_file=json_file_path, benchmark_version="multitarget")

#%%

print("Done!")
