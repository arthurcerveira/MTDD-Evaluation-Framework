from pathlib import Path
from typing import List, Optional
import os
import joblib
import torch

from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize_list, canonicalize
from joblib import delayed

from .rnn_generator import SmilesRnnMoleculeGenerator
from .rnn_utils import load_rnn_model


class SmilesRnnDirectedGenerator(GoalDirectedGenerator):
    def __init__(self, pretrained_model_path: str, n_epochs=4, mols_to_sample=1028, keep_top=512,
                 optimize_n_epochs=2, max_len=100, optimize_batch_size=64, number_final_samples=1028,
                 sample_final_model_only=False, random_start=False, smi_file=None, n_jobs=-1, batch_size=500) -> None:
        self.pretrained_model_path = pretrained_model_path
        self.n_epochs = n_epochs
        self.mols_to_sample = mols_to_sample
        self.keep_top = keep_top
        self.optimize_batch_size = optimize_batch_size
        self.optimize_n_epochs = optimize_n_epochs
        self.pretrain_n_epochs = 0
        self.max_len = max_len
        self.number_final_samples = number_final_samples
        self.sample_final_model_only = sample_final_model_only
        self.random_start = random_start
        self.smi_file = smi_file
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.batch_size = batch_size

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def top_k(self, smiles, scoring_function, k):
        # Score molecules in batches to improve performance
        joblist = (delayed(scoring_function.score_list)(smiles[i:i + self.batch_size]) for i in
                     range(0, len(smiles), self.batch_size))

        scores = self.pool(joblist)
        scores = [score for sublist in scores for score in sublist]

        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None, benchmark_name=None) -> List[str]:

        # fetch initial population?
        if starting_population is None:
            print('selecting initial population...')
            if self.random_start:
                starting_population = []
            else:
                current_path = os.path.dirname(os.path.realpath(__file__))
                top_k_smiles_folder = os.path.join(current_path, '..', '..', 'guacamol', 'data', 'top_k')
                top_k_smiles_path = os.path.join(top_k_smiles_folder, f'{benchmark_name}.smiles')

                if os.path.isfile(top_k_smiles_path):
                    print(f'Loading top k smiles for {benchmark_name} from {top_k_smiles_path}')
                    starting_population = self.load_smiles_from_file(top_k_smiles_path)
                    starting_population = starting_population[:self.mols_to_sample]

                else:
                    print('No top k smiles found, running top k on all smiles')
                    all_smiles = self.load_smiles_from_file(self.smi_file)
                    starting_population = self.top_k(all_smiles, scoring_function, self.mols_to_sample)

        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        model_def = Path(self.pretrained_model_path).with_suffix('.json')

        model = load_rnn_model(model_def, self.pretrained_model_path, device, copy_to_cpu=True)

        generator = SmilesRnnMoleculeGenerator(model=model,
                                               max_len=self.max_len,
                                               device=device)

        molecules = generator.optimise(objective=scoring_function,
                                       start_population=starting_population,
                                       n_epochs=self.n_epochs,
                                       mols_to_sample=self.mols_to_sample,
                                       keep_top=self.keep_top,
                                       optimize_batch_size=self.optimize_batch_size,
                                       optimize_n_epochs=self.optimize_n_epochs,
                                       pretrain_n_epochs=self.pretrain_n_epochs)

        # take the molecules seen during the hill-climbing, and also sample from the final model
        samples = [m.smiles for m in molecules]
        if self.sample_final_model_only:
            samples.clear()
        samples += generator.sample(max(number_molecules, self.number_final_samples))

        # calculate the scores and return the best ones
        samples = canonicalize_list(samples)
        scores = scoring_function.score_list(samples)

        scored_molecules = zip(samples, scores)
        sorted_scored_molecules = sorted(scored_molecules, key=lambda x: (x[1], hash(x[0])), reverse=True)

        top_scored_molecules = sorted_scored_molecules[:number_molecules]

        return [x[0] for x in top_scored_molecules]
