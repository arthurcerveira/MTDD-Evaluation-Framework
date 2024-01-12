from joblib import Parallel, delayed
from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.utils.chemistry import canonicalize
from datetime import datetime
from tqdm import tqdm

pool = Parallel(n_jobs=-1, timeout=10_000)
smi_file = 'guacamol/data/chembl24_canon_train.smiles'
batch_size = 200


def load_smiles_from_file(smi_file):
    with open(smi_file) as f:
        return pool(delayed(canonicalize)(s.strip()) for s in f)


def top_k(smiles, scoring_function, k):
    # Score molecules in batches to improve performance
    joblist = (delayed(scoring_function.score_list)(smiles[i:i + batch_size]) for i in
                    tqdm(range(0, len(smiles), batch_size)))

    scores = pool(joblist)
    scores = [score for sublist in scores for score in sublist]

    scored_smiles = list(zip(scores, smiles))
    scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
    return [smile for score, smile in scored_smiles][:k]


if __name__ == '__main__':
    print(f"[{datetime.now():%H:%M:%S}] Loading SMILES from {smi_file}")
    all_smiles = load_smiles_from_file(smi_file)

    benchmarks = goal_directed_benchmark_suite(version_name="multitarget")

    for i, benchmark in enumerate(benchmarks, 1):
        print(f'[{datetime.now():%H:%M:%S}] Running top k for {i}/{len(benchmarks)}: {benchmark.name}')
        scoring_function = benchmark.wrapped_objective

        top_k_smiles = top_k(all_smiles, scoring_function, 50_000)

        with open(f"guacamol/data/top_k/{benchmark.name}.smiles", 'w') as f:
            f.write('\n'.join(top_k_smiles))

        print(f'[{datetime.now():%H:%M:%S}] Finished top k for {i}/{len(benchmarks)}: {benchmark.name}')

    print(f"[{datetime.now():%H:%M:%S}] Finished running top k for all benchmarks")
