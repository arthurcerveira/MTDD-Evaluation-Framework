import heapq
from typing import List, Optional, Tuple
import os

import joblib
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize
from joblib import delayed

from .chembl_file_reader import ChemblFileReader


class BestFromChemblOptimizer(GoalDirectedGenerator):
    """
    Goal-directed molecule generator that will simply look for the most adequate molecules present in a file.
    """

    def __init__(self, smiles_reader: ChemblFileReader, n_jobs: int, batch_size: int = 500) -> None:
        # self.pool = joblib.Parallel(n_jobs=n_jobs, timeout=10_000)
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        # get a list of all the smiles
        self.smiles = [s for s in smiles_reader]
        self.batch_size = batch_size

    def top_k(self, smiles, scoring_function, k):
        # Score molecules in batches to improve performance
        joblist = (delayed(scoring_function.score_list)(smiles[i:i + self.batch_size]) for i in
                     range(0, len(smiles), self.batch_size))

        scores = self.pool(joblist)
        scores = [score for sublist in scores for score in sublist]

        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None, benchmark_name=None) -> List[str]:
        """
        Will iterate through the reference set of SMILES strings and select the best molecules.
        """
        current_path = os.path.dirname(os.path.realpath(__file__))
        top_k_smiles_folder = os.path.join(current_path, '..', '..', 'guacamol', 'data', 'top_k')
        top_k_smiles_path = os.path.join(top_k_smiles_folder, f'{benchmark_name}.smiles')

        if os.path.isfile(top_k_smiles_path):
            print(f'Loading top k smiles for {benchmark_name} from {top_k_smiles_path}')
            top_k_smi = self.load_smiles_from_file(top_k_smiles_path)
            top_k_smi = top_k_smi[:number_molecules]

        else:
            print('No top k smiles found, running top k on all smiles')
            top_k_smi = self.top_k(self.smiles, scoring_function, number_molecules)

        return top_k_smi
